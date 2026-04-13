import pickle
from cs336_basics.BytePairEncoding import BPETrainer
import time

def main() -> None:
    trainer = BPETrainer(["<|endoftext|>"])
    vocab, merges = trainer.train(
        input_path="data/owt_train.txt",
        vocab_size=32000,
    )
    
    output_path = "./model/bpe_openwebtext_vocab_merges.pkl"

    with open(output_path, "wb") as f:
        pickle.dump((vocab, merges), f)
    print(f"Vocabulary and merges saved to {output_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")