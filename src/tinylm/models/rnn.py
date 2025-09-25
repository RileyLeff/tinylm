from __future__ import annotations

"""Character-level RNN language model implemented with PyTorch."""

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
class RNNConfig:
    vocab_size: int
    embed_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 1
    seq_len: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-2
    grad_clip: float = 1.0
    dropout: float = 0.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    device: str = "auto"
    seed: int = 42
    lr_decay_steps: int | None = None
    lr_decay_gamma: float = 0.5

    def to_dict(self) -> Dict[str, int | float | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, int | float | str]) -> "RNNConfig":
        return cls(**payload)


class CharRNN(nn.Module):
    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.rnn = nn.RNN(
            config.embed_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embed(tokens)
        outputs, hidden = self.rnn(embeddings, hidden)
        outputs = self.output_dropout(outputs)
        logits = self.head(outputs)
        return logits, hidden


def train_language_model(
    model: CharRNN,
    data: CharData,
    *,
    steps: int,
    eval_interval: int,
    rng: np.random.Generator,
    on_eval: Callable[[Dict[str, float]], None] | None = None,
) -> TrainResult:
    config = model.config
    torch.manual_seed(config.seed)

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

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch_x)
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
            val_ppl = evaluate_perplexity(model, data.val_tokens, device)
            record = {
                "step": float(step),
                "mean_loss": mean_loss,
                "last_loss": float(loss.item()),
                "grad_norm": float(grad_norm_value),
                "val_ppl": float(val_ppl),
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

    return TrainResult(metrics=metrics, best_state=best_state, best_step=best_step, best_metric=best_val if best_val < float("inf") else None)


@torch.no_grad()
def evaluate_perplexity(model: CharRNN, tokens: np.ndarray, device: torch.device) -> float:
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
    model.to(device)

    for i in range(0, inputs.shape[0], batch_size):
        batch_x = torch.from_numpy(inputs[i : i + batch_size]).to(device)
        batch_y = torch.from_numpy(targets[i : i + batch_size]).to(device)
        logits, _ = model(batch_x)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            batch_y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_count += batch_y.numel()

    mean_loss = total_loss / total_count
    return math.exp(mean_loss)


@torch.no_grad()
def sample_text(
    model: CharRNN,
    vocab: CharVocab,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int | None,
) -> str:
    config = model.config
    device = resolve_device(config.device)
    model.to(device)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    if prompt:
        prompt_ids = vocab.encode(prompt)
    else:
        rng = np.random.default_rng(seed)
        prompt_ids = np.array([rng.integers(0, vocab.size)], dtype=np.int64)

    tokens = torch.from_numpy(prompt_ids.astype(np.int64)).unsqueeze(0).to(device)
    generated = prompt_ids.tolist()

    hidden: torch.Tensor | None = None
    if tokens.shape[1] > 1:
        for idx in range(tokens.shape[1] - 1):
            _, hidden = model(tokens[:, idx : idx + 1], hidden)

    last_token = tokens[:, -1:]
    for _ in range(max_new_tokens):
        logits, hidden = model(last_token, hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(int(next_token.item()))
        last_token = next_token

    return vocab.decode(np.array(generated, dtype=np.int64))


class RNNModelHandler:
    """Adapter that exposes the RNN through a unified CLI-facing interface."""

    model_id = "rnn"
    config_cls = RNNConfig
    state_filename = "model.pt"
    best_state_filename = "best_model.pt"

    def build_config(self, vocab_size: int, **overrides) -> RNNConfig:
        config = RNNConfig(vocab_size=vocab_size)
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)  # type: ignore[arg-type]
        return config

    def config_to_dict(self, config: RNNConfig) -> Dict[str, int | float | str]:
        return config.to_dict()

    def config_from_dict(self, payload: Dict[str, int | float | str]) -> RNNConfig:
        return RNNConfig.from_dict(payload)

    def create_model(self, config: RNNConfig) -> CharRNN:
        return CharRNN(config)

    def train(
        self,
        model: CharRNN,
        data: CharData,
        *,
        steps: int,
        eval_interval: int,
        rng: np.random.Generator,
        on_eval: Callable[[Dict[str, float]], None] | None = None,
    ) -> TrainResult:
        return train_language_model(
            model,
            data,
            steps=steps,
            eval_interval=eval_interval,
            rng=rng,
            on_eval=on_eval,
        )

    def save_state(self, model: CharRNN, path: Path) -> None:
        torch.save(model.state_dict(), path)

    def save_state_dict(self, state_dict: Dict[str, torch.Tensor], path: Path) -> None:
        torch.save(state_dict, path)

    def load_state(self, model: CharRNN, path: Path) -> None:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)

    def sample(
        self,
        model: CharRNN,
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
    "rnn": RNNModelHandler(),
}
