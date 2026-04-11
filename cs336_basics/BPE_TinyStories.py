import pickle
from cs336_basics.BytePairEncoding import BPETrainer

def main() -> None:
    trainer = BPETrainer(["<|endoftext|>"])
    vocab, merges = trainer.train(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
    )
    
    output_path = "../model/bpe_tinystories_vocab_merges.pkl"

    with open(output_path, "wb") as f:
        pickle.dump((vocab, merges), f)
    print(f"Vocabulary and merges saved to {output_path}")


if __name__ == "__main__":
    main()