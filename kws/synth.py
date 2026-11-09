"""Synthetic "words" made of tone patterns, so demos and tests need no recordings.

Each word is a sequence of segments (fundamental frequency in Hz, relative
duration). Every utterance is rendered with random speaking rate, pitch shift,
amplitude, harmonic balance, leading/trailing silence and additive noise, which
gives DTW realistic time-warping and spectral variation to cope with.
"""

from __future__ import annotations

import numpy as np

VOCABULARY: dict[str, list[tuple[float, float]]] = {
    "yes": [(300, 1.0), (600, 1.0)],
    "no": [(500, 2.0)],
    "up": [(250, 1.0), (400, 1.0), (650, 1.0)],
    "down": [(650, 1.0), (400, 1.0), (250, 1.0)],
    "left": [(350, 1.0), (900, 0.5), (350, 1.0)],
    "right": [(800, 1.5), (450, 1.0)],
    "stop": [(1000, 0.6), (200, 1.4)],
    "go": [(450, 1.0), (450, 0.3), (700, 1.0)],
}


def synth_word(
    word: str,
    sample_rate: int = 16000,
    rng: np.random.Generator | None = None,
    base_duration: float = 0.45,
    snr_db: float | None = 20.0,
    variation: float = 1.0,
) -> np.ndarray:
    """Render one utterance of ``word``.

    ``variation`` scales all random perturbations (0 = identical every time).
    ``snr_db=None`` disables noise.
    """
    if word not in VOCABULARY:
        raise KeyError(f"unknown word {word!r}; choose from {sorted(VOCABULARY)}")
    rng = np.random.default_rng() if rng is None else rng
    segments = VOCABULARY[word]
    rate = np.exp(rng.uniform(-0.25, 0.25) * variation)  # speaking-rate factor
    pitch = np.exp(rng.uniform(-0.04, 0.04) * variation)  # +-4 % pitch shift
    h2 = 0.5 + rng.uniform(-0.2, 0.2) * variation  # 2nd harmonic level
    h3 = 0.25 + rng.uniform(-0.1, 0.1) * variation  # 3rd harmonic level
    total = sum(d for _, d in segments)

    # Instantaneous frequency track with short glides between segments.
    freq_track = []
    for f, d in segments:
        n = max(1, int(sample_rate * base_duration * rate * d / total))
        seg_jitter = np.exp(rng.uniform(-0.02, 0.02) * variation)
        freq_track.append(np.full(n, f * pitch * seg_jitter))
    freq = np.concatenate(freq_track)
    glide = max(1, int(0.02 * sample_rate))
    freq = np.convolve(np.pad(freq, glide, mode="edge"), np.ones(glide) / glide, mode="same")
    freq = freq[glide:-glide]

    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    tone = np.sin(phase) + h2 * np.sin(2 * phase) + h3 * np.sin(3 * phase)
    attack = min(len(tone) // 4, int(0.03 * sample_rate))
    envelope = np.ones(len(tone))
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[len(tone) - attack :] = np.linspace(1, 0, attack)
    voice = tone * envelope * rng.uniform(0.4, 0.9)

    lead = np.zeros(int(sample_rate * rng.uniform(0.05, 0.2)))
    tail = np.zeros(int(sample_rate * rng.uniform(0.05, 0.2)))
    signal = np.concatenate([lead, voice, tail])
    if snr_db is not None:
        p_signal = np.mean(voice**2)
        noise_std = np.sqrt(p_signal / (10 ** (snr_db / 10)))
        signal = signal + rng.normal(0, noise_std, size=signal.shape)
    return signal.astype(np.float32)


def make_dataset(
    words: list[str] | None = None,
    n_per_word: int = 10,
    seed: int = 0,
    sample_rate: int = 16000,
    snr_db: float | None = 20.0,
) -> list[tuple[str, np.ndarray]]:
    """Generate ``n_per_word`` utterances for every word, reproducibly."""
    rng = np.random.default_rng(seed)
    words = list(VOCABULARY) if words is None else words
    return [
        (w, synth_word(w, sample_rate, rng, snr_db=snr_db))
        for w in words
        for _ in range(n_per_word)
    ]
