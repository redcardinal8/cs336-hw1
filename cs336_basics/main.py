from cs336_basics.train_bpe_tinystories import train_bpe_tinystories
from cs336_basics.train_bpe_expts_owt import train_bpe_expts_owt


if __name__ == "__main__":
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    path = "data/owt_train.txt"
    vocab, merges = train_bpe_expts_owt(path, vocab_size, special_tokens)