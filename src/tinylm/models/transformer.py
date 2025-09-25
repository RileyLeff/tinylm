from __future__ import annotations

"""Minimal transformer language model implemented with PyTorch."""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tinylm.data import CharData, CharVocab, batch_iterator
from tinylm.models.common import TrainResult


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TransformerConfig:
    vocab_size: int
    seq_len: int = 256
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    device: str = "auto"
    seed: int = 42
    batch_size: int = 64
    lr_decay_steps: int | None = None
    lr_decay_gamma: float = 0.5

    def to_dict(self) -> Dict[str, int | float | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, int | float | str]) -> "TransformerConfig":
        return cls(**payload)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attention = attn_weights @ v
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attention)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config.embed_dim, config.num_heads, config.dropout)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.ln2(x))
        return x + self.dropout(ff_out)


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Embedding(config.seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model maximum ({self.config.seq_len}). "
                "Increase config.seq_len or provide shorter sequences."
            )
        device = tokens.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_embed(positions)
        x = self.dropout(x)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        device = tokens.device
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.seq_len :]
            logits = self.forward(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def train_transformer_language_model(
    model: TransformerLM,
    data: CharData,
    *,
    steps: int,
    eval_interval: int,
    rng: np.random.Generator | None,
    on_eval: Callable[[Dict[str, float]], None] | None = None,
) -> TrainResult:
    config = model.config
    torch.manual_seed(config.seed)
    if rng is None:
        rng = np.random.default_rng(config.seed)

    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.lr_decay_steps is not None and config.lr_decay_steps > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_decay_steps,
            gamma=config.lr_decay_gamma,
        )

    iterator = batch_iterator(data.train_tokens, config.seq_len, config.batch_size, rng=rng)
    metrics: List[Dict[str, float]] = []
    losses: List[float] = []
    grad_norm_value = 0.0
    interval_start = time.perf_counter()

    best_state: Dict[str, torch.Tensor] | None = None
    best_step: float | None = None
    best_val: float = float("inf")

    for step in range(1, steps + 1):
        model.train()
        batch_x, batch_y = next(iterator)
        batch_x = torch.from_numpy(batch_x).to(device)
        batch_y = torch.from_numpy(batch_y).to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            batch_y.view(-1),
            reduction="mean",
        )
        loss.backward()
        grad_norm_value = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item())
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.item()))

        if step % eval_interval == 0 or step == steps:
            duration = time.perf_counter() - interval_start
            mean_loss = float(np.mean(losses)) if losses else float(loss.item())
            val_ppl = float(evaluate_perplexity(model, data.val_tokens, device))
            record = {
                "step": float(step),
                "mean_loss": mean_loss,
                "last_loss": float(loss.item()),
                "grad_norm": float(grad_norm_value),
                "val_ppl": val_ppl,
                "interval_seconds": float(duration),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            metrics.append(record)
            if on_eval is not None:
                on_eval(record)
            if val_ppl < best_val:
                best_val = float(val_ppl)
                best_step = float(step)
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            losses.clear()
            interval_start = time.perf_counter()

    return TrainResult(
        metrics=metrics,
        best_state=best_state,
        best_step=best_step,
        best_metric=best_val if best_val < float("inf") else None,
    )


@torch.no_grad()
def evaluate_perplexity(model: TransformerLM, tokens: np.ndarray, device: torch.device) -> float:
    config = model.config
    seq_len = config.seq_len
    batch_size = config.batch_size

    total_tokens = len(tokens) - 1
    if total_tokens <= 0:
        raise ValueError("Not enough tokens to evaluate")

    tokens_per_batch = seq_len * batch_size
    num_batches = total_tokens // tokens_per_batch
    if num_batches == 0:
        batch_size = 1
        tokens_per_batch = seq_len * batch_size
        num_batches = total_tokens // tokens_per_batch
    if num_batches == 0:
        raise ValueError("Validation corpus is too small for evaluation")

    usable_tokens = num_batches * tokens_per_batch
    trimmed = tokens[: usable_tokens + 1]
    inputs = trimmed[:-1].reshape(num_batches * batch_size, seq_len)
    targets = trimmed[1:].reshape(num_batches * batch_size, seq_len)

    total_loss = 0.0
    total_count = 0

    model.eval()
    for i in range(0, inputs.shape[0], batch_size):
        batch_x = torch.from_numpy(inputs[i : i + batch_size]).to(device)
        batch_y = torch.from_numpy(targets[i : i + batch_size]).to(device)
        logits = model(batch_x)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            batch_y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_count += batch_x.numel()

    mean_loss = total_loss / total_count
    return math.exp(mean_loss)


@torch.no_grad()
def sample_text(
    model: TransformerLM,
    vocab: CharVocab,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int | None,
) -> str:
    config = model.config
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()

    if prompt:
        prompt_ids = vocab.encode(prompt)
    else:
        prompt_ids = np.array([np.random.randint(0, vocab.size)], dtype=np.int64)

    tokens = torch.from_numpy(prompt_ids.astype(np.int64)).unsqueeze(0).to(device)
    generated = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=temperature)
    return vocab.decode(generated.squeeze(0).cpu().numpy())


class TransformerModelHandler:
    model_id = "transformer"
    state_filename = "model.pt"
    best_state_filename = "best_model.pt"

    def build_config(self, vocab_size: int, **overrides) -> TransformerConfig:
        config = TransformerConfig(vocab_size=vocab_size)
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    def config_to_dict(self, config: TransformerConfig) -> Dict[str, int | float | str]:
        return config.to_dict()

    def config_from_dict(self, payload: Dict[str, int | float | str]) -> TransformerConfig:
        return TransformerConfig.from_dict(payload)

    def create_model(self, config: TransformerConfig) -> TransformerLM:
        return TransformerLM(config)

    def train(
        self,
        model: TransformerLM,
        data: CharData,
        *,
        steps: int,
        eval_interval: int,
        rng: np.random.Generator,
        on_eval: Callable[[Dict[str, float]], None] | None = None,
    ) -> TrainResult:
        return train_transformer_language_model(
            model,
            data,
            steps=steps,
            eval_interval=eval_interval,
            rng=rng,
            on_eval=on_eval,
        )

    def save_state(self, model: TransformerLM, path: Path) -> None:
        torch.save(model.state_dict(), path)

    def save_state_dict(self, state_dict: Dict[str, torch.Tensor], path: Path) -> None:
        torch.save(state_dict, path)

    def load_state(self, model: TransformerLM, path: Path) -> None:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)

    def sample(
        self,
        model: TransformerLM,
        vocab: CharVocab,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> str:
        return sample_text(
            model,
            vocab,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )

    def state_path(self, run_dir: Path) -> Path:
        return run_dir / self.state_filename

    def best_state_path(self, run_dir: Path) -> Path:
        return run_dir / self.best_state_filename


MODEL_REGISTRY = {
    "transformer": TransformerModelHandler(),
}
