import pickle
from cs336_basics.BPE_Ryan import train_bpe
import time

def main() -> None:
    # trainer = BPETrainer(["<|endoftext|>"])
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        split_special_token="<|endoftext|>",
        num_processes=8,
    )
    
    output_path = "./model/bpe_tinystories_vocab_merges.pkl"

    with open(output_path, "wb") as f:
        pickle.dump((vocab, merges), f)
    print(f"Vocabulary and merges saved to {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")