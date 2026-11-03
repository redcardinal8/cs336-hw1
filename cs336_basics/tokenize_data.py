"""
Tokenize a text corpus into a flat uint16 array on disk (readable with np.memmap).

    uv run python -m cs336_basics.tokenize_data \
        --input data/TinyStoriesV2-GPT4-train.txt --output data/tinystories_train.bin \
        --vocab tinystories_bpe_out/vocab.pkl --merges tinystories_bpe_out/merges.pkl
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time

import numpy as np

from cs336_basics.find_chunk_boundaries import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

_tokenizer: Tokenizer | None = None


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]):
    global _tokenizer
    _tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def _encode_range(args: tuple[str, int, int]) -> np.ndarray:
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.asarray(_tokenizer.encode(text), dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--num-chunks", type=int, default=512, help="more chunks = less memory per worker")
    args = parser.parse_args()

    probe = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    assert max(probe.vocab) < 2**16, "vocab too large for uint16"

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_chunks, [s.encode() for s in args.special_tokens])
    jobs = [(args.input, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    t0 = time.time()
    n_bytes = os.path.getsize(args.input)
    n_tokens = 0
    # Stream chunk results to disk in order so we never hold the whole token array in memory.
    with open(args.output, "wb") as out, multiprocessing.Pool(
        args.num_workers, initializer=_init_worker, initargs=(args.vocab, args.merges, args.special_tokens)
    ) as pool:
        for i, ids in enumerate(pool.imap(_encode_range, jobs)):
            ids.tofile(out)
            n_tokens += len(ids)
            if (i + 1) % 16 == 0 or i + 1 == len(jobs):
                el = time.time() - t0
                print(f"[{i + 1}/{len(jobs)}] {n_tokens:,} tokens, {el:.0f}s", flush=True)

    el = time.time() - t0
    print(f"Done: {n_tokens:,} tokens -> {args.output}")
    print(f"Compression: {n_bytes / max(n_tokens, 1):.3f} bytes/token, throughput {n_bytes / el / 1e6:.2f} MB/s")


if __name__ == "__main__":
    main()
