import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

class BPETokenizer:
    def __init__(self,
        special_tokens: list[str]
        ):
        """
        Initialize the tokenizer with:
         - initial vocabulary: special tokens + the 256 bytes
         - initial vocabulary size: 256 + number of special tokens
         - merge records: empty
        """
        self.special_tokens_count = len(special_tokens)
        self.special_tokens = special_tokens
        self.curr_vocab_size: int = 256 + self.special_tokens_count
        self.merges: list[tuple[bytes, bytes]] = []

        # initialise vocabulary
        self.vocab: dict[int, bytes] = {}
        for i in range(self.special_tokens_count):
            self.vocab[i] = special_tokens[i].encode
        for i in range(256):
            self.vocab[i + self.special_tokens_count] = chr(i)

        # initialise helper dicts: words_counter, word_to_tokens
        self.words_counter: dict[str, int] = Counter()
        self.word_to_tokens: dict[str, list[int]] = {}
    
    def train(self,
        input_path: str | os.PathLike,
        vocab_size: int,
        **kwargs,
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train the BPE tokenizer.
        """
        # tokenise with the initial vocab to get words_counter and word_to_tokens
        self.init_tokenization(input_path)

        # count all the token pairs
        token_pairs_counts = defaultdict(int)
        for word, word_count in self.words_counter.items():
            word_tokens = self.word_to_tokens[word]
            word_token_pairs = zip(word_tokens[:-1], word_tokens[1:])
            word_token_pairs_counts = Counter(word_token_pairs)
            for pair, pair_count in word_token_pairs_counts.items():
                token_pairs_counts[pair] += word_count * pair_count
        
        # find the pair with max count and max alphanumerical order
        merge_pair = max(
            token_pairs_counts,
            key=lambda key: (token_pairs_counts.get(key), key)
            )
        
        while self.curr_vocab_size < vocab_size: # perform merge actions.
            # update vocab, vocab size, merges, token pair counts
            self.vocab[self.curr_vocab_size] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]
            new_token_id = self.curr_vocab_size
            self.curr_vocab_size += 1
            self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))
            token_pairs_counts.pop(merge_pair)
            
            # merge: for each word:
            # - change word tokens
            # - for each newly formed pairs, add to new pair, subtract from old pair
            for word, word_count in self.words_counter.items():
                word_tokens = self.word_to_tokens[word]
                new_word_tokens = word_tokens.copy()
                word_merges = 0
                for i in range(len(word_tokens) - 1):
                    if tuple(word_tokens[i:i+2]) == merge_pair:
                        # edit new word tokens
                        new_word_tokens = new_word_tokens[:i-word_merges] \
                            + [new_token_id] \
                            + new_word_tokens[i-word_merges+2:]
                        # edit token pair counts
                        if i < len(word_tokens) - 2:
                            token_pairs_counts[tuple(word_tokens[i+1:i+3])] -= word_count
                            token_pairs_counts[(new_token_id, word_tokens[i+2])] += word_count
                        if i > 0:
                            token_pairs_counts[tuple(word_tokens[i-1:i+1])] -= word_count
                            token_pairs_counts[(word_tokens[i-1], new_token_id)] += word_count
                        # edit word merges count
                        word_merges += 1
                self.word_to_tokens[word] = new_word_tokens

            # find the next pair with max count and max alphanumerical order
            # TODO: use caching to avoid iterating over all byte pairs
            merge_pair = max(
                token_pairs_counts,
                key=lambda key: (token_pairs_counts.get(key), key)
                )
        
        return self.vocab, self.merges

    def init_tokenization(self,
            input_path: str | os.PathLike,
            split_special_token: bytes = b"<|endoftext|>",
        ) -> None:
        """
        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            special_tokens (list[str]): List of special tokens.
        
        Returns nothing, but updates:
            self.words_counter: counts the number of occurences of each word
            self.word_to_tokens: returns the initial tokenization of each word
        
        First - pretokenization:
         - split on <|endoftext|> special tokens into num_processes segments
         - pretokenize each segment to get segment_words_counter
         - combine the results to get words_counter
        
        Then - tokenization:
         - for word in words_counter, tokenize each word to get word_to_tokens
        """

        with open(input_path, "rb") as f:
            num_processes = 4
            print(f)
            # split on <|endoftext|> special tokens
            boundaries = self.find_chunk_boundaries(f, num_processes, split_special_token)
            special_tokens = self.special_tokens

            # The following is a serial implementation
            # TODO: parallelize this by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                segment_words_counter = self.pretokenize(chunk, special_tokens)
                self.words_counter += segment_words_counter
            

            # The following is a parallel implementation
            tasks = zip(boundaries[:-1], boundaries[1:])
        
        for word in self.words_counter:
            self.word_to_tokens[word] = list(int(i) + self.special_tokens_count for i in word.encode())

    @staticmethod
    def find_chunk_boundaries( # from pretokenization_example.py
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
        ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        assert mini_chunk_size > len(split_special_token), "Mini chunk size must be larger than the special token"

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size - len(split_special_token) + 1

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def pretokenize(
        text: str,
        special_tokens: list
        ) -> dict[str, int]:
        """
        Pretokenize the input text into a list of pre-tokens,
        and return local_words_counter.
        """
        # initialise words_counter and word_to_tokens
        local_words_counter: dict[str, int] = defaultdict(int)

        # split on special tokens, removing them from the text
        SPE_PAT = '|'.join([re.escape(token) for token in special_tokens])
        split_text = re.split(SPE_PAT, text)

        # pretokenize each segment to get segment_words_counter
        TOK_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""    
        for segment in split_text:
            iter = re.finditer(TOK_PAT, segment)
            for match in iter:
                word = match.group(0)
                local_words_counter[word] += 1
        return Counter(local_words_counter)

    def simplified_train(self,
        input_str: str,
        target_vocab_size: int,
        **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Simplified example: split by whitespace instead of regex, no special tokens
        """

        # count unique words in input string
        """def init_tokenization(self,
            input_path: str | os.PathLike,
        ) -> tuple[dict[str, int], dict[str, list[int]]]

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
        
        Returns:
            (words_counter, word_to_tokens)
            words_counter: counts the number of occurences of each word
            word_to_tokens: returns the initial tokenization of each word
        
        First - pretokenization:
         - add special tokens to words_counter
         - split on special tokens
         - pretokenize each segment to get segment_words_counter
         - combine the results to get words_counter
        
        Then - tokenization:
         - for word in words_counter, tokenize each word to get word_to_tokens

        Return words_counter and word_to_tokens
        """
        words_list = input_str.split(" ")
        words_counter = Counter(words_list)

        # map word to encoding (no special tokens)
        word_to_tokens: dict[str, list[int]] = {}
        for word in words_counter:
            word_to_tokens[word] = list(word.encode())

        # count all the token pairs
        token_pairs_counts = defaultdict(int)
        for word, word_count in words_counter.items():
            word_tokens = word_to_tokens[word]
            word_token_pairs = zip(word_tokens[:-1], word_tokens[1:])
            word_token_pairs_counts = Counter(word_token_pairs)
            for pair, pair_count in word_token_pairs_counts.items():
                token_pairs_counts[pair] += word_count * pair_count
        
        # find the pair with max count and max alphanumerical order
        merge_pair = max(
            token_pairs_counts,
            key=lambda key: (token_pairs_counts.get(key), key)
            )
        
        while self.curr_vocab_size < target_vocab_size: # perform merge actions.
            
            # update vocab, vocab size, merges, token pair counts
            self.vocab[self.curr_vocab_size] = self.vocab[merge_pair[0]] + self.vocab[merge_pair[1]]
            new_token_id = self.curr_vocab_size
            self.curr_vocab_size += 1
            self.merges.append((self.vocab[merge_pair[0]], self.vocab[merge_pair[1]]))
            token_pairs_counts.pop(merge_pair)
            
            # merge: for each word:
            # - change word tokens
            # - for each newly formed pairs, add to new pair, subtract from old pair
            for word, word_count in words_counter.items():
                word_tokens = word_to_tokens[word]
                new_word_tokens = word_tokens.copy()
                word_merges = 0
                for i in range(len(word_tokens) - 1):
                    if tuple(word_tokens[i:i+2]) == merge_pair:
                        # edit new word tokens
                        new_word_tokens = new_word_tokens[:i-word_merges] \
                            + [new_token_id] \
                            + new_word_tokens[i-word_merges+2:]
                        # edit token pair counts
                        if i < len(word_tokens) - 2:
                            token_pairs_counts[tuple(word_tokens[i+1:i+3])] -= word_count
                            token_pairs_counts[(new_token_id, word_tokens[i+2])] += word_count
                        if i > 0:
                            token_pairs_counts[tuple(word_tokens[i-1:i+1])] -= word_count
                            token_pairs_counts[(word_tokens[i-1], new_token_id)] += word_count
                        # edit word merges count
                        word_merges += 1
                word_to_tokens[word] = new_word_tokens

            # find the next pair with max count and max alphanumerical order
            # TODO: use caching to avoid iterating over all byte pairs
            merge_pair = max(
                token_pairs_counts,
                key=lambda key: (token_pairs_counts.get(key), key)
                )
        
        return self.vocab, self.merges

                
    # def train(self,
    #     input_path: str | os.PathLike,
    #     target_vocab_size: int,
    #     special_tokens: list[str],
    #     **kwargs,
    # ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #     """
    #     Given the path to an input corpus, run train a BPE tokenizer and
    #     output its vocabulary and merges.

    #     Args:
    #         input_path (str | os.PathLike): Path to BPE tokenizer training data.
    #         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
    #         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
    #             These strings will never be split into multiple tokens, and will always be
    #             kept as a single token. If these special tokens occur in the `input_path`,
    #             they are treated as any other string.

    #     Returns:
    #         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #             vocab:
    #                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
    #                 to bytes (token bytes)
    #             merges:
    #                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
    #                 representing that <token1> was merged with <token2>.
    #                 Merges are ordered by order of creation.
    #     """
    #     # add special tokens to vocab list
    #     for i, token in enumerate(special_tokens):
    #         self.vocab[256+i] = token
    #         self.curr_vocab_size += 1

# tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
# vocab_size = 300

# file_path = "../tests/fixtures/tinystories_sample.txt"
# with open(file_path, "rb") as f:
#     sample = f.read()
#     vocab, merges = tokenizer.train(input_path=file_path, vocab_size=vocab_size)
#     print(tokenizer.words_counter)
#     for tok in tokenizer.word_to_tokens[' unexpected']:
#         print(tokenizer.vocab[tok])
        
# tokenizer = BPETokenizer(special_tokens=["<|endoftext|>"])
# merge_count = 1000

# file_path = "../tests/fixtures/tinystories_sample.txt"
# with open(file_path, "rb") as f:
#     sample = f.read()
#     tokenizer.init_tokenization(file_path)
#     print(tokenizer.words_counter)
#     for tok in tokenizer.word_to_tokens[' unexpected']:
#         print(tokenizer.vocab[tok])

# tokenizer = BPETokenizer(special_tokens=[])
# merge_count = 10
# sample = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
# vocab, merges = tokenizer.simplified_train(
#     input_str=sample.decode(),
#     target_vocab_size=256+merge_count,
# )
# print(merges)


# def run_train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """Given the path to an input corpus, run train a BPE tokenizer and
#     output its vocabulary and merges.

#     Args:
#         input_path (str | os.PathLike): Path to BPE tokenizer training data.
#         vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
#         special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
#             These strings will never be split into multiple tokens, and will always be
#             kept as a single token. If these special tokens occur in the `input_path`,
#             they are treated as any other string.

#     Returns:
#         tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#             vocab:
#                 The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
#                 to bytes (token bytes)
#             merges:
#                 BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
#                 representing that <token1> was merged with <token2>.
#                 Merges are ordered by order of creation.
#     """
#     raise NotImplementedError