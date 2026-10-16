import json
from dataclasses import asdict

import numpy as np
import torch

from model import ModelConfig, StoryModel
from prepare_data import digest
from prepare_tokens import OUT, verify


def main():
    if torch.__version__ != "2.10.0+cpu":
        raise RuntimeError("Use the course CPU PyTorch installation.")
    if np.__version__ != "2.4.6":
        raise RuntimeError("Install numpy==2.4.6.")
    verify()
    token_manifest = json.loads(
        (OUT / "manifest.json").read_text(encoding="utf-8")
    )
    config = ModelConfig()
    if token_manifest["vocab_size"] != config.vocab_size:
        raise RuntimeError("Tokenizer and model vocabulary differ.")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = StoryModel(config).cpu()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 2_557_888:
        raise RuntimeError("Unexpected course parameter count.")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError("Input and output weights are not shared.")

    required = 2 * (config.context_length + 1)
    values = np.fromfile(
        OUT / "train.bin", dtype="<u2", count=required
    )
    if values.size != required:
        raise RuntimeError("Not enough training tokens for inspection.")
    sequences = torch.from_numpy(
        values.astype(np.int64)
    ).reshape(2, config.context_length + 1)
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]

    model.eval()
    with torch.no_grad():
        embeddings = model.token_embedding(inputs)
        logits, loss = model(inputs, targets)
        expected_shape = (2, 256, 4096)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError("Unexpected logit shape.")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits.")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss.")

        cut = config.context_length // 2
        changed = inputs.clone()
        changed[:, cut:] = (
            changed[:, cut:] + 1
        ) % config.vocab_size
        altered, _ = model(changed)
        torch.testing.assert_close(
            logits[:, :cut], altered[:, :cut],
            rtol=1e-5, atol=1e-6,
        )
        causal_difference = (
            logits[:, :cut] - altered[:, :cut]
        ).abs().max().item()

    model.train()
    model.zero_grad(set_to_none=True)
    _, backward_loss = model(inputs, targets)
    backward_loss.backward()
    gradient_norms = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
        gradient_norms.append(parameter.grad.norm())
    gradient_norm = torch.stack(gradient_norms).norm().item()
    if gradient_norm == 0:
        raise RuntimeError("All gradients are zero.")
    model.zero_grad(set_to_none=True)

    try:
        model(torch.zeros(1, 257, dtype=torch.long))
    except ValueError:
        pass
    else:
        raise RuntimeError("Context limit was not enforced.")

    print("Configuration:", json.dumps(asdict(config), indent=2))
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Token manifest SHA256:", digest(OUT / "manifest.json"))
    print("Parameters:", f"{parameters:,}")
    print("Input shape:", list(inputs.shape))
    print("Embedding shape:", list(embeddings.shape))
    print("Logit shape:", list(logits.shape))
    print("Initial loss:", f"{loss.item():.4f}")
    print("Causal check maximum difference:", causal_difference)
    print("Gradient norm:", f"{gradient_norm:.4f}")
    print("Shared weights: passed")
    print("Causal mask: passed")
    print("Gradient check: passed")
    print("Context limit: passed")
    print("Model inspection passed.")


if __name__ == "__main__":
    main()
