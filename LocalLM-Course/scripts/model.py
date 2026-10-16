from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 256
    width: int = 192
    layers: int = 4
    heads: int = 6
    hidden_width: int = 512
    norm_eps: float = 1e-5
    rope_base: float = 10000.0

    def __post_init__(self):
        dimensions = (
            self.vocab_size, self.context_length, self.width,
            self.layers, self.heads, self.hidden_width,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Model dimensions must be positive.")
        if self.width % self.heads != 0:
            raise ValueError("Width must be divisible by heads.")
        if (self.width // self.heads) % 2 != 0:
            raise ValueError("Rotary positions need even head width.")
        if self.norm_eps <= 0 or self.rope_base <= 0:
            raise ValueError(
                "Normalization and rotary values must be positive."
            )


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(mean_square + self.eps)
        return (values * self.weight.float()).to(x.dtype)


class RotaryPositions(nn.Module):
    def __init__(self, head_width, context_length, base):
        super().__init__()
        powers = torch.arange(0, head_width, 2).float()
        inverse_frequencies = base ** (-powers / head_width)
        positions = torch.arange(context_length).float()
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cosines", angles.cos(), persistent=False)
        self.register_buffer("sines", angles.sin(), persistent=False)

    def forward(self, x):
        length = x.shape[-2]
        cosines = self.cosines[:length].to(x.dtype)[None, None]
        sines = self.sines[:length].to(x.dtype)[None, None]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack(
            (even * cosines - odd * sines,
             even * sines + odd * cosines),
            dim=-1,
        )
        return rotated.flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(
            config.width, 3 * config.width, bias=False
        )
        self.query_norm = RMSNorm(self.head_width, config.norm_eps)
        self.key_norm = RMSNorm(self.head_width, config.norm_eps)
        self.positions = RotaryPositions(
            self.head_width, config.context_length, config.rope_base
        )
        self.output = nn.Linear(
            config.width, config.width, bias=False
        )

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.positions(self.query_norm(q))
        k = self.positions(self.key_norm(k))
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch, length, width)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(
            config.width, config.hidden_width, bias=False
        )
        self.up = nn.Linear(
            config.width, config.hidden_width, bias=False
        )
        self.down = nn.Linear(
            config.hidden_width, config.width, bias=False
        )

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.norm_eps)
        self.attention = CausalAttention(config)
        self.feedforward_norm = RMSNorm(config.width, config.norm_eps)
        self.feedforward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class StoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.width
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layers)
        )
        self.final_norm = RMSNorm(config.width, config.norm_eps)
        self.lm_head = nn.Linear(
            config.width, config.vocab_size, bias=False
        )
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError(
                "Input must be a [batch, time] long tensor."
            )
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.config.context_length:
            raise ValueError(
                "Input batch or sequence length is invalid."
            )
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("Targets must match the input shape.")
            if targets.dtype != torch.long:
                raise ValueError("Targets must use torch.long.")

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss
