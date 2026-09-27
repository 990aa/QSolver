"""Pure numerical operations used by the confidence-aware ensemble."""

import numpy as np


def sharpen_probabilities(probabilities: np.ndarray, power: float = 1.5) -> np.ndarray:
    """Increase separation between options while preserving row-wise normalization."""
    if power <= 0:
        raise ValueError("power must be positive")
    powered = np.power(np.asarray(probabilities, dtype=float), power)
    totals = powered.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Each probability row must contain a positive value")
    return powered / totals


def percentile_ranks(probabilities: np.ndarray) -> np.ndarray:
    """Convert each row into normalized ranks in the interval (0, 1]."""
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, values.shape[1] + 1)
    return ranks / values.shape[1]


def confidence_weights(deberta_probabilities: np.ndarray) -> np.ndarray:
    """Return per-row DeBERTa, scratch, and decoder weights from top-two margin."""
    sorted_values = np.sort(deberta_probabilities, axis=1)[:, ::-1]
    margins = sorted_values[:, 0] - sorted_values[:, 1]
    weights = np.empty((len(margins), 3), dtype=float)
    weights[margins >= 0.35] = (0.82, 0.11, 0.07)
    medium = (margins >= 0.15) & (margins < 0.35)
    weights[medium] = (0.62, 0.23, 0.15)
    weights[margins < 0.15] = (0.42, 0.34, 0.24)
    return weights


def blend_predictions(deberta: np.ndarray, scratch: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    """Blend three probability matrices using calibrated and rank-based signals."""
    if not (deberta.shape == scratch.shape == decoder.shape):
        raise ValueError("All model probability matrices must have the same shape")
    weights = confidence_weights(deberta)
    sharpened = np.stack(
        [sharpen_probabilities(deberta, 1.8), sharpen_probabilities(scratch), sharpen_probabilities(decoder)],
        axis=1,
    )
    ranked = np.stack([percentile_ranks(deberta), percentile_ranks(scratch), percentile_ranks(decoder)], axis=1)
    return 0.85 * np.sum(weights[:, :, None] * sharpened, axis=1) + 0.15 * np.sum(
        weights[:, :, None] * ranked, axis=1
    )