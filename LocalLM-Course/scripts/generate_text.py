import argparse
from datetime import datetime, timezone
import math
import sys
import time
import uuid

from inference import ROOT, digest, environment, generate
from inference import load_model, project_path, save_report


def new_report(directory):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"generation-{stamp}-{uuid.uuid4().hex[:8]}.json"
    return project_path(directory) / name


def run_prompt(model, tokenizer, metadata, args, prompt, load_seconds):
    output = (
        project_path(args.output) if args.output
        else new_report(args.output_dir)
    )
    if output.exists():
        raise FileExistsError("Choose an unused output filename.")
    samples = []
    print("Generating...", flush=True)
    started = time.perf_counter()
    for index in range(args.num_samples):
        sample = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + index,
        )
        samples.append(sample)
    elapsed = time.perf_counter() - started
    result = {
        **metadata,
        "protocol": "local-story-generation-v1",
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": args.num_samples,
        "load_seconds": load_seconds,
        "generation_seconds": elapsed,
        "environment": environment(),
        "source_sha256": {
            name: digest(ROOT / "scripts" / name)
            for name in ("generate_text.py", "inference.py", "model.py")
        },
        "samples": samples,
    }
    save_report(output, result)
    for index, sample in enumerate(samples, start=1):
        print(f"\nSample {index}, seed {sample['seed']}")
        print(sample["text"])
        print("Finish:", sample["finish_reason"])
        print("New tokens:", sample["generated_tokens"])
    print(f"Generation seconds: {elapsed:.3f}")
    print("Saved:", output)


def main():
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--tokenizer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt")
    mode.add_argument("--prompt-file")
    mode.add_argument("--interactive", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive.")
    if args.max_new_tokens < 1 or args.num_samples < 1:
        parser.error("Token and sample counts must be positive.")
    if args.seed < 0 or args.seed + args.num_samples > 2**32:
        parser.error("Sample seeds must be unsigned 32-bit integers.")
    if args.interactive and args.output:
        parser.error("Use --output-dir for an interactive session.")
    if args.output and project_path(args.output).exists():
        parser.error("Choose an unused output filename.")
    started = time.perf_counter()
    model, tokenizer, metadata = load_model(
        args.checkpoint, args.tokenizer
    )
    load_seconds = time.perf_counter() - started
    if not 1 <= args.top_k <= model.config.vocab_size:
        parser.error("top-k must be between 1 and vocabulary size.")
    parameters = sum(item.numel() for item in model.parameters())
    print("Checkpoint:", metadata["checkpoint_path"])
    print("Completed updates:", metadata["completed_updates"])
    print(f"Parameters: {parameters:,}")
    print("Context tokens:", model.config.context_length)
    print("Device: cpu | dtype: float32 | threads: 4")
    print(f"Load seconds: {load_seconds:.3f}", flush=True)
    if args.interactive:
        print("Enter a story beginning. Type /quit to exit.")
        print("Each prompt starts fresh with the configured seeds.")
        while True:
            try:
                prompt = input("\nStory beginning: ").strip()
            except EOFError:
                break
            if prompt == "/quit":
                break
            if not prompt:
                continue
            try:
                run_prompt(
                    model, tokenizer, metadata, args,
                    prompt, load_seconds
                )
            except ValueError as error:
                print("Input error:", error)
    else:
        prompt = (
            project_path(args.prompt_file).read_text(
                encoding="utf-8-sig"
            )
            if args.prompt_file else args.prompt
        )
        run_prompt(
            model, tokenizer, metadata, args, prompt, load_seconds
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. No model weights were changed.")
        raise SystemExit(130)
