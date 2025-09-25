from __future__ import annotations

"""Model registry for tinylm CLI."""

from typing import Dict

from .rnn import MODEL_REGISTRY as RNN_MODELS
from .transformer import MODEL_REGISTRY as TRANSFORMER_MODELS

MODEL_REGISTRY: Dict[str, object] = {}
MODEL_REGISTRY.update(RNN_MODELS)
MODEL_REGISTRY.update(TRANSFORMER_MODELS)

__all__ = ["MODEL_REGISTRY"]
