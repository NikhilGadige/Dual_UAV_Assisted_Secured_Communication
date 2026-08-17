"""
Loads the SINR -> semantic-similarity lookup table produced by training the
actual DeepSC model (train_deepsc.py -> deepsc_lookup_table.csv) and exposes
it as an interpolation function.

This module is intentionally torch-free: the Week 9 ISAC system model calls
semantic_similarity_from_sinr_db() at simulation time and should not need to
load PyTorch or the DeepSC network itself for every SINR evaluation. Run
train_deepsc.py once (or whenever the corpus/architecture changes) to
regenerate the CSV this module reads.
"""

import csv
import os
from typing import List, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_TABLE_PATH = os.path.join(THIS_DIR, "deepsc_lookup_table.csv")

_cached_table: Tuple[List[float], List[float], List[float]] = None


def _load_table() -> Tuple[List[float], List[float], List[float]]:
    global _cached_table
    if _cached_table is not None:
        return _cached_table

    if not os.path.exists(LOOKUP_TABLE_PATH):
        raise FileNotFoundError(
            f"DeepSC lookup table not found at {LOOKUP_TABLE_PATH}. "
            "Run `python train_deepsc.py` from the Week9 work/ directory to "
            "train the DeepSC model and generate it."
        )

    sinr_db, similarity, accuracy = [], [], []
    with open(LOOKUP_TABLE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            sinr_db.append(float(row["sinr_db"]))
            similarity.append(float(row["semantic_similarity"]))
            accuracy.append(float(row["word_accuracy"]))

    order = sorted(range(len(sinr_db)), key=lambda i: sinr_db[i])
    sinr_db = [sinr_db[i] for i in order]
    similarity = [similarity[i] for i in order]
    accuracy = [accuracy[i] for i in order]

    _cached_table = (sinr_db, similarity, accuracy)
    return _cached_table


def _interp(x: float, xs: List[float], ys: List[float]) -> float:
    """Linear interpolation with flat extrapolation outside the table range
    (matches numpy.interp's default behavior, without requiring numpy)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def semantic_similarity_from_sinr_db(sinr_db: float) -> float:
    """DeepSC-derived semantic similarity S(SINR) in [0, 1], interpolated
    from the trained-model evaluation table (replaces the hand-tuned
    logistic approximation)."""
    table_sinr_db, similarity, _ = _load_table()
    return _interp(sinr_db, table_sinr_db, similarity)


def word_accuracy_from_sinr_db(sinr_db: float) -> float:
    """DeepSC-derived word-level reconstruction accuracy at a given SINR,
    interpolated from the trained-model evaluation table (diagnostic only;
    semantic_similarity_from_sinr_db is what feeds the system model)."""
    table_sinr_db, _, accuracy = _load_table()
    return _interp(sinr_db, table_sinr_db, accuracy)
