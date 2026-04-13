import pickle
from cs336_basics.BytePairEncoding import BPETrainer
import time

def main() -> None:
    trainer = BPETrainer(["<|endoftext|>"])
    vocab, merges = trainer.train(
        input_path="data/owt_train.txt",
        vocab_size=32000,
    )
    
    vocab_path = "./model/bpe_openwebtext_vocab.pkl"
    merges_path = "./model/bpe_openwebtext_merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"Vocabulary and merges saved to {vocab_path} and {merges_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")