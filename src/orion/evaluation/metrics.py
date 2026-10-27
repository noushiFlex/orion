"""Probabilistic scores for 3-way outcomes (0 home, 1 draw, 2 away). Lower is better except accuracy."""

import numpy as np


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1))))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def rps(y: np.ndarray, p: np.ndarray) -> float:
    """Ranked probability score: respects the order home < draw < away."""
    cum_p, cum_o = np.cumsum(p, axis=1)[:, :2], np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / 2))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y))


def evaluate(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return {"log_loss": log_loss(y, p), "brier": brier(y, p), "rps": rps(y, p), "accuracy": accuracy(y, p), "n": len(y)}
