"""Isolated-word recognition by DTW nearest-template matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import trim_silence
from .dtw import dtw
from .mfcc import deltas, mfcc


def extract_features(signal: np.ndarray, sample_rate: int = 16000, use_deltas: bool = True):
    """Trim silence, then MFCC (+ deltas) -> array of shape (frames, 13 or 26)."""
    feats = mfcc(trim_silence(signal, sample_rate), sample_rate)
    if use_deltas:
        feats = np.hstack([feats, deltas(feats)])
    return feats


@dataclass
class Result:
    label: str | None
    distance: float
    scores: dict[str, float] = field(default_factory=dict)


class TemplateRecognizer:
    """Store MFCC templates per label and classify by smallest DTW distance."""

    def __init__(self, sample_rate: int = 16000, band: int | None = None, threshold=None):
        self.sample_rate = sample_rate
        self.band = band
        self.threshold = threshold  # reject if best distance exceeds this
        self.templates: list[tuple[str, np.ndarray]] = []

    @property
    def labels(self) -> list[str]:
        return sorted({label for label, _ in self.templates})

    def enroll(self, label: str, signal: np.ndarray) -> None:
        self.templates.append((label, extract_features(signal, self.sample_rate)))

    def recognize(self, signal: np.ndarray) -> Result:
        if not self.templates:
            raise RuntimeError("no templates enrolled")
        feats = extract_features(signal, self.sample_rate)
        scores: dict[str, float] = {}
        for label, template in self.templates:
            d = dtw(feats, template, band=self.band)
            scores[label] = min(d, scores.get(label, np.inf))
        best = min(scores, key=scores.get)
        if self.threshold is not None and scores[best] > self.threshold:
            return Result(None, scores[best], scores)
        return Result(best, scores[best], scores)

    def save(self, path: str | Path) -> None:
        arrays = {f"t{i}": feats for i, (_, feats) in enumerate(self.templates)}
        labels = np.array([label for label, _ in self.templates])
        np.savez(path, labels=labels, sample_rate=self.sample_rate, **arrays)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> TemplateRecognizer:
        with np.load(path, allow_pickle=False) as data:
            rec = cls(sample_rate=int(data["sample_rate"]), **kwargs)
            rec.templates = [(str(lbl), data[f"t{i}"]) for i, lbl in enumerate(data["labels"])]
        return rec


def evaluate(recognizer: TemplateRecognizer, dataset) -> tuple[float, dict]:
    """Return (accuracy, confusion dict {(true, predicted): count})."""
    confusion: dict[tuple[str, str | None], int] = {}
    correct = 0
    for label, signal in dataset:
        pred = recognizer.recognize(signal).label
        confusion[(label, pred)] = confusion.get((label, pred), 0) + 1
        correct += pred == label
    return correct / len(dataset), confusion
