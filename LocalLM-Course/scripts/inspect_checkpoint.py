import argparse

import torch

from model import ModelConfig, StoryModel
from prepare_data import ROOT, digest
from training_utils import TrainingConfig, make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    path = ROOT / args.checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format_version"] != 1:
        raise RuntimeError("Unsupported checkpoint format.")
    values = dict(saved["config"])
    values["model"] = ModelConfig(**values["model"])
    config = TrainingConfig(**values)
    completed = saved["completed_updates"]
    if type(completed) is not int or not 0 <= completed <= (
        config.max_updates
    ):
        raise RuntimeError("Invalid completed update count.")

    torch.set_num_threads(config.threads)
    model = StoryModel(config.model).to(
        device="cpu", dtype=torch.float32
    )
    model.load_state_dict(saved["model"], strict=True)
    optimizer = make_optimizer(model, config)
    optimizer.load_state_dict(saved["optimizer"])
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Non-finite saved weights.")
    if completed > 0:
        if len(optimizer.state) != len(list(model.parameters())):
            raise RuntimeError("Missing optimizer state.")
        for state in optimizer.state.values():
            if state["step"].item() != completed:
                raise RuntimeError("Optimizer update count differs.")
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(
                    value
                ).all():
                    raise RuntimeError("Non-finite optimizer state.")
    for key in ("torch_rng_state", "train_rng_state"):
        torch.Generator(device="cpu").set_state(saved[key])

    print("Checkpoint:", path)
    print("SHA256:", digest(path))
    print("Completed updates:", completed)
    print("Configured update budget:", config.max_updates)
    print("Last validation update:", saved["last_val_update"])
    print("Last validation loss:", saved["last_val"])
    print("Model, optimizer, and random states: readable")
    print("Checkpoint inspection passed.")


if __name__ == "__main__":
    main()
