import argparse
import sys

from inference import ROOT, digest, environment, generate
from inference import load_model, project_path, save_report


PROMPTS = (
    "A little dog found a red ball near the gate.",
    "Mia wanted to help her friend, but she was afraid.",
    "The small boat stopped beside a quiet island.",
    "Ben opened the box and saw something surprising.",
    "A bird lost its way while flying home.",
)


def main():
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if project_path(args.output).exists():
        raise FileExistsError("Choose an unused report filename.")
    model, tokenizer, metadata = load_model(
        args.checkpoint, args.tokenizer
    )
    samples = []
    for index, prompt in enumerate(PROMPTS):
        seed = 42 + index
        sample = generate(
            model, tokenizer, prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            seed=seed,
        )
        samples.append(sample)
        print(f"\nPrompt {index + 1}, seed {seed}: {prompt}")
        print("Continuation:", sample["continuation"])
        print("Finish:", sample["finish_reason"], flush=True)
    result = {
        **metadata,
        "protocol": "fixed-story-prompts-v1",
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "environment": environment(),
        "source_sha256": {
            name: digest(ROOT / "scripts" / name)
            for name in (
                "review_samples.py", "inference.py", "model.py"
            )
        },
        "samples": samples,
    }
    save_report(args.output, result)
    print("Report:", project_path(args.output))


if __name__ == "__main__":
    main()
