import argparse
import json
from pathlib import Path

import torch

from cs336_basics.generation import generate, load_model
from cs336_basics.tokenizer import Tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [args.eos_token])
    model = load_model(args.checkpoint, args.config, args.device)
    vocab_size = model.lm_head.weight.shape[0]
    if set(tokenizer.vocab) != set(range(vocab_size)):
        raise ValueError("Tokenizer IDs must exactly match the model vocabulary")
    prompt_ids = tokenizer.encode(args.prompt)
    eos_id = tokenizer.token_to_id[args.eos_token.encode("utf-8")]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    stats = {}
    ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        eos_id,
        generator,
        use_kv_cache=args.kv_cache,
        stats=stats,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "kv_cache": args.kv_cache,
        "timing": stats,
        "generated_token_ids": ids[len(prompt_ids) :],
        "generated_tokens": len(ids) - len(prompt_ids),
        "stopped_on_eos": len(ids) > len(prompt_ids) and ids[-1] == eos_id,
        "text": tokenizer.decode(ids),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(report["text"])


if __name__ == "__main__":
    main()
