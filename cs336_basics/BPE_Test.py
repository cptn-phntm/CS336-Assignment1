import pickle
from cs336_basics.BytePairEncoding import BPETrainer
import time

def main() -> None:
    trainer = BPETrainer(["<|endoftext|>"])
    vocab, merges = trainer.train(
        input_path="data/test.txt",
        vocab_size=257+20,
    )
    
    print(merges)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")