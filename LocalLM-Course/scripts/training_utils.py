import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig
from prepare_data import ROOT
from prepare_tokens import OUT


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    seed: int
    threads: int
    micro_batch_size: int
    accumulation_steps: int
    max_updates: int
    learning_rate: float
    final_learning_rate: float
    decay_after_update: int
    beta1: float
    beta2: float
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float
    log_interval: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    keep_checkpoints: int
    model: ModelConfig

    def __post_init__(self):
        if not isinstance(self.run_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_name
        ):
            raise ValueError("Use letters, digits, - or _ in run_name.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")

        integer_fields = (
            "threads", "micro_batch_size", "accumulation_steps",
            "max_updates", "decay_after_update", "log_interval",
            "eval_interval", "eval_batches", "checkpoint_interval",
            "keep_checkpoints",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        positive_fields = (
            "learning_rate", "final_learning_rate",
            "adam_epsilon", "max_grad_norm",
        )
        numeric_fields = positive_fields + (
            "weight_decay", "beta1", "beta2",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(
                value
            ):
                raise ValueError(f"{name} must be a finite number.")
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1).")
        if self.final_learning_rate > self.learning_rate:
            raise ValueError("Final learning rate exceeds initial.")
        if self.decay_after_update > self.max_updates:
            raise ValueError("Decay point exceeds the update budget.")
        if asdict(self.model) != asdict(ModelConfig()):
            raise ValueError("Use the Lesson 7 model for the baseline.")

    @property
    def run_dir(self):
        return ROOT / "models" / self.run_name

    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.accumulation_steps

    @property
    def tokens_per_update(self):
        return self.effective_batch_size * self.model.context_length


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    values["model"] = ModelConfig(**values["model"])
    return TrainingConfig(**values)


class TokenBatches:
    def __init__(self, split, batch_size, context_length, seed):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown data split.")
        self.tokens = np.memmap(
            OUT / f"{split}.bin", dtype="<u2", mode="r"
        )
        self.batch_size = batch_size
        self.context_length = context_length
        if self.tokens.size <= context_length:
            raise ValueError("Not enough tokens for a complete window.")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next(self):
        starts = torch.randint(
            self.tokens.size - self.context_length,
            (self.batch_size,), generator=self.generator,
        ).numpy()
        offsets = np.arange(self.context_length + 1)
        indices = starts[:, None] + offsets[None, :]
        rows = torch.from_numpy(
            self.tokens[indices].astype(np.int64)
        )
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def learning_rate_for(config, update):
    if not 1 <= update <= config.max_updates:
        raise ValueError("Update number is outside the run budget.")
    if update <= config.decay_after_update:
        return config.learning_rate
    return config.final_learning_rate


def make_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )


def run_update(model, optimizer, batches, config, update):
    model.train()
    rate = learning_rate_for(config, update)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    mean_loss = 0.0

    for _ in range(config.accumulation_steps):
        inputs, targets = batches.next()
        logits, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        mean_loss += loss.detach().item() / config.accumulation_steps
        (loss / config.accumulation_steps).backward()
        del logits, loss

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.max_grad_norm,
        norm_type=2.0, error_if_nonfinite=True, foreach=False,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": mean_loss,
        "grad_norm": gradient_norm.item(),
        "lr": rate,
    }
