from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TrainResult:
    metrics: List[Dict[str, float]]
    best_state: Optional[Dict[str, Any]]
    best_step: Optional[float]
    best_metric: Optional[float]
