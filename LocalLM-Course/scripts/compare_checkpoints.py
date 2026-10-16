import argparse
from pathlib import Path

import torch

from prepare_data import ROOT


def load(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format_version"] != 1:
        raise RuntimeError("Unsupported checkpoint format.")
    return saved


def compare(left, right, label):
    if torch.is_tensor(left):
        if not torch.is_tensor(right):
            raise RuntimeError(f"Different value types: {label}")
        if left.dtype != right.dtype or left.shape != right.shape:
            raise RuntimeError(f"Different tensor structure: {label}")
        if not torch.equal(left, right):
            raise RuntimeError(f"Different tensor values: {label}")
        return
    if type(left) is not type(right):
        raise RuntimeError(f"Different value types: {label}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"Different dictionary keys: {label}")
        for key in left:
            compare(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"Different sequence lengths: {label}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{label}[{index}]")
    elif left != right:
        raise RuntimeError(f"Different values: {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)

    first_config = dict(reference["config"])
    second_config = dict(candidate["config"])
    first_config.pop("run_name")
    second_config.pop("run_name")
    compare(first_config, second_config, "config")
    compare(reference["fingerprint"], candidate["fingerprint"],
            "fingerprint")
    compare(reference["completed_updates"],
            candidate["completed_updates"], "completed_updates")

    for key in (
        "model", "optimizer", "torch_rng_state", "train_rng_state"
    ):
        compare(reference[key], candidate[key], key)
        print(f"{key}: exactly equal")

    print("Completed updates:", reference["completed_updates"])
    print("Training state comparison passed.")


if __name__ == "__main__":
    main()
