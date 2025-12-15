import collections
import math
from typing import Dict, List, Tuple

import numpy as np


class RunningStat:
    """Welford's algorithm for mean and variance (online)."""

    def __init__(self, eps: float = 1e-8):
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0
        self.eps: float = eps
        self.max: float = -np.inf
        self.min: float = np.inf

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.max = max(self.max, x)
        self.min = min(self.min, x)

    @property
    def var(self) -> float:
        return (self.M2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.n > 1 else self.eps


class RewardBalancer:
    """
    Maintain running stats for each sub-reward and produce normalized values
    and dynamic weights.
    """

    def __init__(
        self,
        keys: list[str],
        eps: float = 1e-8,
        temp: float = 1.0,
        min_weight: float = 0.01,
    ):
        # keys: list of sub-reward names (strings)
        self.stats: Dict[str, RunningStat] = {k: RunningStat(eps=eps) for k in keys}
        self.eps = eps
        self.temp = temp
        self.min_weight = min_weight
        self.keys = list(keys)

    def update_stats(self, values: Dict[str, float]):
        for k, v in values.items():
            self.stats[k].update(v)

    def normalized(self, values: Dict[str, float]) -> Dict[str, float]:
        """Return z-score like normalization: value / (std + eps)."""
        out = {}
        for k, v in values.items():
            std = self.stats[k].std + self.eps
            out[k] = v / std
        return out

    def inverse_variance_weights(self) -> Dict[str, float]:
        """Return weights proportional to 1/std (inverse variance), then normalized."""
        inv: List[float] = []
        for k in self.keys:
            s = self.stats[k].std + self.eps
            inv.append(1.0 / s)
        inv_np = np.array(inv)
        w = inv_np / (inv_np.sum() + 1e-12)
        # enforce min weight and renormalize
        w = np.maximum(w, self.min_weight)
        w = w / w.sum()
        return dict(zip(self.keys, w.tolist()))

    def softmax_of_neglogstd(self) -> Dict[str, float]:
        """Softmax weighting using -log(std) (bigger weight -> smaller std)."""
        scores: List[float] = []
        for k in self.keys:
            s = self.stats[k].std + self.eps
            scores.append(-math.log(s + self.eps))
        # temperature softmax
        scores_np = np.array(scores) / max(self.temp, 1e-12)
        exp = np.exp(scores_np - scores_np.max())
        w = exp / exp.sum()
        w = np.maximum(w, self.min_weight)
        w = w / w.sum()
        return dict(zip(self.keys, w.tolist()))

    def dynamic_weighted_sum(
        self, values: Dict[str, float], mode: str = "softmax"
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        values: raw sub-term values dict
        mode: 'softmax' or 'invvar' or 'uniform'
        returns: (weighted_sum, weights_dict, normalized_values)
        """
        self.update_stats(values)
        norm_values = self.normalized(values)  # scale-invariant contributions

        if mode == "softmax":
            weights = self.softmax_of_neglogstd()
        elif mode == "invvar":
            weights = self.inverse_variance_weights()
        else:
            weights = {k: 1.0 / len(self.keys) for k in self.keys}

        total = 0.0
        for k in self.keys:
            # you can choose to use norm_values[k] or raw values[k]
            total += weights[k] * norm_values[k]
        return total, weights, norm_values
