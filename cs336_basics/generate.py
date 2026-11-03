"""
Sample text from a trained checkpoint.

    uv run python -m cs336_basics.generate --run-dir runs/ts_base \
        --vocab tinystories_bpe_out/vocab.pkl --merges tinystories_bpe_out/merges.pkl \
        --prompt "Once upon a time" --temperature 0.8 --top-p 0.9 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.serialization import load_checkpoint
from cs336_basics.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        context = ids[:, -model.context_length :]
        logits = model(context)[0, -1]
        if temperature <= 0:
            next_id = int(logits.argmax())
        else:
            probs = softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cumulative = sorted_probs.cumsum(-1)
                # Keep the smallest prefix whose mass reaches top_p (always keep the top token).
                keep = cumulative - sorted_probs < top_p
                sorted_probs = sorted_probs * keep
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()
            next_id = int(torch.multinomial(probs, 1))
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if eos_id is not None and next_id == eos_id:
            break
    return ids[0, len(prompt_ids) :].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <run-dir>/ckpt_latest.pt")
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = json.loads((Path(args.run_dir) / "config.json").read_text())
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg["rope_theta"],
        use_rope=cfg["use_rope"],
        use_norm=cfg["use_norm"],
        norm_position=cfg["norm_position"],
        ffn_type=cfg["ffn_type"],
    )
    load_checkpoint(args.checkpoint or Path(args.run_dir) / "ckpt_latest.pt", model)
    model.to(args.device)

    special = "<|endoftext|>"
    tok = Tokenizer.from_files(args.vocab, args.merges, [special])
    eos_id = tok.byte_to_id[special.encode()]
    out = generate(
        model,
        tok.encode(args.prompt),
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
        device=args.device,
    )
    print(args.prompt + tok.decode(out))
    print(f"\n[{len(out)} tokens generated]")


if __name__ == "__main__":
    main()
