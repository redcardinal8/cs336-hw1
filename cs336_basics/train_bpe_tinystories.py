import time
import pickle
import resource  # Unix/Mac only; use psutil on Windows
from pathlib import Path

from cs336_basics.train_bpe import train_bpe

def train_bpe_tinystories(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    out_dir: str = "tinystories_bpe_out"
): 
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- time + memory tracking ---
    start_time = time.time()
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    end_time = time.time()
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    elapsed = end_time - start_time
    peak_mem_mb = mem_after / 1024  

    print(f"Training took {elapsed / 60:.2f} minutes ({elapsed:.1f} s)")
    print(f"Peak memory usage: {peak_mem_mb:.1f} MB")

    # --- serialize vocab and merges ---
    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # human-readable versions too, since vocab is int -> bytes
    with open(out_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for idx, tok_bytes in sorted(vocab.items()):
            f.write(f"{idx}\t{tok_bytes!r}\n")

    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a!r} {b!r}\n")

    # --- longest token ---
    longest_id, longest_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(f"Longest token (id={longest_id}, len={len(longest_bytes)} bytes): {longest_bytes!r}")
    try:
        print(f"  decoded: {longest_bytes.decode('utf-8')!r}")
    except UnicodeDecodeError:
        print("  (not valid standalone UTF-8)")

    return vocab, merges

