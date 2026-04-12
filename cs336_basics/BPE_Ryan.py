import regex as re
from collections import defaultdict, Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
import heapq
from multiprocessing import Pool


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# PAT = r"\S+"

# used for bytes comparison in the heap
class Reverse:
    def __init__(self, val: tuple[bytes, bytes]):
        self.val = val
    def __lt__(self, other):
        return self.val > other.val


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str | None = None,
    num_processes: int | None = 3,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert special_tokens
    assert vocab_size >= 256 + len(special_tokens)
    
    # initialize the vocabulary
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    
    if vocab_size == len(vocab):
        return vocab, []
    
    # pretokenization

    if not split_special_token:
        split_special_token = special_tokens[0]
    if num_processes:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token=split_special_token.encode("utf-8"))
        assert boundaries
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker, [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
        assert results
        freq_table: Counter[tuple[bytes, ...]] = Counter()
        pair_count: Counter[tuple[bytes, bytes]] = Counter()
        pair_to_pretoken: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        for result in results:
            freq_table += result[0]
            pair_count += result[1]
            for pair, pretoken_set in result[2].items():
                pair_to_pretoken[pair] = pair_to_pretoken[pair].union(pretoken_set)


    else:
        freq_table, pair_count, pair_to_pretoken = pretokenization(input_path, 0, None, special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    # initialize heap
    pair_count_heap = []
    for k, v in pair_count.items():
        heapq.heappush(pair_count_heap, (-v, Reverse(k)))

    
    # tokenize
    while len(vocab) < vocab_size:

        # print(pair_count)
        
        if not pair_count:
            raise ValueError("desired vocab_size is infeasible")
        
        # get the pair to merge at this step
        count, max_pair = heapq.heappop(pair_count_heap)
        max_pair = max_pair.val
        while pair_count[max_pair] != -count:
            # stale data in heap
            count, max_pair = heapq.heappop(pair_count_heap)
            max_pair = max_pair.val
        max_pair_merged = max_pair[0] + max_pair[1]
        # iterate and merge, maintain freq_table, pair_count, and pair_to_pretoken
        pairs_to_update: set[tuple[bytes, bytes]] = set()
        for pretoken_tuple in pair_to_pretoken[max_pair].copy():
            merged_pretoken_tuple = merge(pretoken_tuple, max_pair)
            # freq_table update
            freq = freq_table.pop(pretoken_tuple)
            freq_table[merged_pretoken_tuple] = freq
            # delete old pair_count and pair_to_pretoken
            for i in range(len(pretoken_tuple)-1):
                pair = (pretoken_tuple[i], pretoken_tuple[i+1])
                pairs_to_update.add(pair)
                pair_count[pair] -= freq
                assert pair_count[pair] >= 0
                pair_to_pretoken[pair].discard(pretoken_tuple)
            # add new merged_pretoken to pair_count and pair_to_pretoken
            for i in range(len(merged_pretoken_tuple)-1):
                pair = (merged_pretoken_tuple[i], merged_pretoken_tuple[i+1])
                pairs_to_update.add(pair)
                pair_count[pair] += freq
                pair_to_pretoken[pair].add(merged_pretoken_tuple)
        merges.append(max_pair)
        for pair in pairs_to_update:
            heapq.heappush(pair_count_heap, (-pair_count[pair], Reverse(pair)))
        vocab[len(vocab)] = max_pair_merged
    
    return vocab, merges


def worker(args):
    return pretokenization(*args)
    

def pretokenization(
    input_path: str,
    start: int,
    end: int | None,
    special_tokens: list[str]
) -> tuple[Counter[tuple[bytes, ...]], Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Compute the freq_table, pair_count, pair_to_pretoken on a specified chunk of the input file. Can be used in parallel.
    """
    freq_table: Counter[tuple[bytes, ...]] = Counter()
    pair_count: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_pretoken: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        length = f.tell()
        f.seek(0)
        if not end:
            end = length
        assert end <= length
        assert start < end
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split on special tokens
        chunk_split = re.split("|".join(re.escape(t) for t in special_tokens), chunk)

        #constuct the frequency table
        for s in chunk_split:
            for match in re.finditer(PAT, s):
                pretoken = match.group().encode("utf-8")
                freq_table[tuple(bytes([b]) for b in pretoken)] += 1
        
        for pretoken_tuple, freq in freq_table.items():
            for i in range(len(pretoken_tuple)-1):
                pair = (pretoken_tuple[i], pretoken_tuple[i+1])
                pair_count[pair] += freq
                pair_to_pretoken[pair].add(pretoken_tuple)
    
    return freq_table, pair_count, pair_to_pretoken


def merge(bytes_tuple: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Returns the bytes_tuple after merging all neighbour merge_pairs.
    """
    n = len(bytes_tuple)
    if n <= 1:
        return bytes_tuple # no merge can be performed
    merged_bytes: list[bytes] = []
    merged_pair = merge_pair[0] + merge_pair[1]
    i = 0
    while i < n:
        if i < n-1 and bytes_tuple[i] == merge_pair[0] and bytes_tuple[i+1] == merge_pair[1]:
            merged_bytes.append(merged_pair)
            i += 2
        else:
            merged_bytes.append(bytes_tuple[i])
            i += 1
    
    return tuple(merged_bytes)
