"""MFCC feature extraction implemented from scratch with NumPy.

Pipeline: pre-emphasis -> framing -> Hamming window -> power spectrum (FFT)
-> triangular mel filterbank -> log -> DCT-II -> (optional) liftering and
cepstral mean normalisation.
"""

from __future__ import annotations

import numpy as np


def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - coeff * x[n-1] (boosts high frequencies)."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def frame_signal(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split a 1-D signal into overlapping frames, zero-padding the tail.

    Returns an array of shape ``(n_frames, frame_len)`` where
    ``n_frames = 1 + ceil(max(0, len(signal) - frame_len) / hop)``.
    """
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    n_frames = 1 + int(np.ceil(max(0, n - frame_len) / hop))
    pad = (n_frames - 1) * hop + frame_len - n
    padded = np.concatenate([signal, np.zeros(max(0, pad))])
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[idx]


def hamming(n: int) -> np.ndarray:
    """Hamming window w[k] = 0.54 - 0.46 cos(2 pi k / (n - 1))."""
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    return 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Periodogram estimate |FFT|^2 / n_fft, shape ``(n_frames, n_fft // 2 + 1)``."""
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (np.abs(spec) ** 2) / n_fft


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_filters: int, n_fft: int, sample_rate: int, f_min: float = 0.0, f_max: float | None = None
) -> np.ndarray:
    """Triangular filters equally spaced on the mel scale.

    Returns shape ``(n_filters, n_fft // 2 + 1)``; each filter peaks at 1.
    """
    f_max = sample_rate / 2 if f_max is None else f_max
    if not 0 <= f_min < f_max <= sample_rate / 2:
        raise ValueError("need 0 <= f_min < f_max <= sample_rate / 2")
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_bins)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bank = np.zeros((n_filters, n_bins))
    for m in range(n_filters):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        bank[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def dct_ii(x: np.ndarray, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II along the last axis, keeping the first ``n_out`` coefficients."""
    n = x.shape[-1]
    k = np.arange(n_out)[:, None]
    i = np.arange(n)[None, :]
    basis = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    scale = np.full(n_out, np.sqrt(2.0 / n))
    scale[0] = np.sqrt(1.0 / n)
    return (x @ basis.T) * scale


def lifter(cepstra: np.ndarray, L: int = 22) -> np.ndarray:
    """Sinusoidal liftering to de-emphasise higher cepstral coefficients."""
    if L <= 0:
        return cepstra
    n = np.arange(cepstra.shape[-1])
    return cepstra * (1 + (L / 2) * np.sin(np.pi * n / L))


def mfcc(
    signal: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_filters: int = 26,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_fft: int | None = None,
    preemph: float = 0.97,
    lifter_L: int = 22,
    cmn: bool = True,
) -> np.ndarray:
    """Compute MFCCs, shape ``(n_frames, n_mfcc)``."""
    if n_mfcc > n_filters:
        raise ValueError("n_mfcc must be <= n_filters")
    frame_len = int(round(sample_rate * frame_ms / 1000))
    hop = int(round(sample_rate * hop_ms / 1000))
    if n_fft is None:
        n_fft = 1 << (frame_len - 1).bit_length()
    frames = frame_signal(pre_emphasis(signal, preemph), frame_len, hop)
    frames = frames * hamming(frame_len)
    pspec = power_spectrum(frames, n_fft)
    energies = pspec @ mel_filterbank(n_filters, n_fft, sample_rate).T
    log_e = np.log(np.maximum(energies, np.finfo(float).eps))
    feats = lifter(dct_ii(log_e, n_mfcc), lifter_L)
    if cmn:
        feats = feats - feats.mean(axis=0, keepdims=True)
    return feats


def deltas(feats: np.ndarray, N: int = 2) -> np.ndarray:
    """Regression-based delta coefficients over +-N frames."""
    if N < 1:
        raise ValueError("N must be >= 1")
    denom = 2 * sum(n * n for n in range(1, N + 1))
    padded = np.pad(feats, ((N, N), (0, 0)), mode="edge")
    T = feats.shape[0]
    out = np.zeros_like(feats, dtype=np.float64)
    for n in range(1, N + 1):
        out += n * (padded[N + n : N + n + T] - padded[N - n : N - n + T])
    return out / denom
