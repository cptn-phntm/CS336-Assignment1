import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict

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
        special_tokens_count = len(special_tokens)
        self.curr_vocab_size: int = 256 + special_tokens_count
        self.merges: list[tuple[bytes, bytes]] = []

        # initialise vocabulary
        self.vocab: dict[int, bytes] = {}
        for i in range(special_tokens_count):
            self.vocab[i] = special_tokens[i]
        for i in range(256):
            self.vocab[i + special_tokens_count] = chr(i)
        

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
        
tokenizer = BPETokenizer(special_tokens=[])
merge_count = 6
sample = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
vocab, merges = tokenizer.simplified_train(
    input_str=sample,
    target_vocab_size=256+merge_count,
)
print(merges)


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