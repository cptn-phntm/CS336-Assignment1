import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

def pretokenize_worker(job):
    input_path, start, end, special_tokens = job
    return pretokenize(input_path, start, end, special_tokens)

def pretokenize(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list
    ) -> dict[str, int]:
    """
    Pretokenize the input text into a list of pre-tokens,
    and return local_words_counter.
    """
    # open the file and read the chunk from start to end, decode as utf-8
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
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

class BPETrainer:
    def __init__(self, special_tokens: list[str]):
        """
        Initialize the tokenizer with:
         - initial vocabulary: special tokens + the 256 bytes
         - initial vocabulary size: 256 + number of special tokens
         - merge records: empty
        """
        self.special_tokens = special_tokens
        self.curr_vocab_size: int = 256 + len(special_tokens)
        self.merges: list[tuple[bytes, bytes]] = []

        # initialise vocabulary
        self.vocab: dict[int, bytes] = {}
        for i in range(256):
            self.vocab[i] = bytes([i])
        for ind, tok in enumerate(special_tokens):
            self.vocab[256 + ind] = tok.encode('utf-8')

    def train(self,
        input_path: str,
        vocab_size: int,
        split_special_token: bytes = b"<|endoftext|>",
        num_processes: int = 4,
        ) -> None:
        """
        Train the tokenizer on the input file until the vocabulary size reaches vocab_size.
        """
        words_counter: dict[str, int] = Counter()
        word_to_tokens: dict[str, list[int]] = {}
        special_tokens = self.special_tokens

        # Step 1: Pretokenize the input file and get the initial words_counter

        # Break up the file into chunks
        with open(input_path, "rb") as f:
            # split on <|endoftext|> special tokens
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
        
        # Pretokenise each chunk and combine the result

        # # The following is a naive implementation that works.
        # with open(input_path, "r", encoding="utf-8") as f:
        #     text = f.read()
        #     words_counter = pretokenize(text, self.special_tokens)

        # # The following is a serial implementation
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     # Run pre-tokenization on your chunk and store the counts for each pre-token
        #     segment_words_counter = pretokenize(input_path, start, end, special_tokens)
        #     words_counter += segment_words_counter
            

        # The following is a parallel implementation
        jobs = [
            (input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for chunk_words_counter in executor.map(pretokenize_worker, jobs):
                words_counter += chunk_words_counter

        # Generate word_to_tokens mapping for each word in words_counter
        for word in words_counter:
            word_to_tokens[word] = list(int(i) for i in word.encode())
        
        # Step 2: Iteratively merge the most common pair of tokens until vocab_size is reached

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
            key=lambda key: (
                token_pairs_counts.get(key),
                self.vocab[key[0]],
                self.vocab[key[1]])
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
            for word, word_count in words_counter.items():
                word_tokens = word_to_tokens[word]
                new_word_tokens = word_tokens.copy()
                word_merges = 0
                i = 0
                while i < len(word_tokens) - 1:
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
                        i += 2
                    else:
                        i += 1
                word_to_tokens[word] = new_word_tokens

            # find the next pair with max count and max alphanumerical order
            # TODO: use caching to avoid iterating over all byte pairs
            merge_pair = max(
                token_pairs_counts,
                key=lambda key: (
                    token_pairs_counts.get(key),
                    self.vocab[key[0]],
                    self.vocab[key[1]])
            )
        
        return self.vocab, self.merges

