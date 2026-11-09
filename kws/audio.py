"""WAV input/output, resampling and energy-based silence trimming."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def to_float_mono(data: np.ndarray) -> np.ndarray:
    """Convert integer or float PCM (any channel count) to float64 mono in [-1, 1]."""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if info.min == 0:  # unsigned 8-bit
            data = (data.astype(np.float64) - 128) / 128
        else:
            data = data.astype(np.float64) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data


def read_wav(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """Read a WAV file as float mono at ``target_rate`` Hz."""
    rate, data = wavfile.read(str(path))
    signal = to_float_mono(data)
    if rate != target_rate:
        g = gcd(rate, target_rate)
        signal = resample_poly(signal, target_rate // g, rate // g)
    return signal


def write_wav(path: str | Path, signal: np.ndarray, sample_rate: int = 16000) -> None:
    """Write float audio as 16-bit PCM."""
    clipped = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    wavfile.write(str(path), sample_rate, (clipped * 32767).astype(np.int16))


def trim_silence(
    signal: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: float = 20.0,
    threshold_db: float = -30.0,
    margin_ms: float = 30.0,
) -> np.ndarray:
    """Cut leading/trailing frames whose energy is ``threshold_db`` below the loudest frame."""
    signal = np.asarray(signal, dtype=np.float64)
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(signal) // frame
    if n_frames == 0:
        return signal
    energy = (signal[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
    peak = energy.max()
    if peak <= 0:
        return signal
    db = 10 * np.log10(np.maximum(energy, 1e-20) / peak)
    active = np.flatnonzero(db > threshold_db)
    margin = int(sample_rate * margin_ms / 1000)
    start = max(0, active[0] * frame - margin)
    end = min(len(signal), (active[-1] + 1) * frame + margin)
    return signal[start:end]
