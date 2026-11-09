import numpy as np
import pytest

from kws import cli
from kws.audio import read_wav, to_float_mono, trim_silence, write_wav
from kws.dtw import dtw
from kws.mfcc import (
    dct_ii,
    deltas,
    frame_signal,
    hamming,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    mfcc,
    pre_emphasis,
)
from kws.recognizer import TemplateRecognizer, evaluate
from kws.synth import VOCABULARY, make_dataset, synth_word

SR = 16000


# ------------------------------------------------------------------ MFCC


def test_pre_emphasis():
    x = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(pre_emphasis(x, 0.97), [1.0, 0.03, 0.03])


@pytest.mark.parametrize("n,expected", [(400, 1), (401, 2), (560, 2), (561, 3), (16000, 99)])
def test_frame_count(n, expected):
    frames = frame_signal(np.zeros(n), 400, 160)
    assert frames.shape == (expected, 400)


def test_frames_content():
    x = np.arange(11.0)
    frames = frame_signal(x, 4, 3)
    assert frames.shape == (4, 4)
    np.testing.assert_array_equal(frames[1], [3, 4, 5, 6])
    np.testing.assert_array_equal(frames[-1], [9, 10, 0, 0])


def test_hamming_matches_numpy():
    np.testing.assert_allclose(hamming(400), np.hamming(400))


def test_mel_roundtrip():
    hz = np.array([0.0, 440.0, 1000.0, 8000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-9)
    assert abs(hz_to_mel(1000.0) - 1000.0) < 1.0


def test_filterbank_shape_and_peaks():
    bank = mel_filterbank(26, 512, SR)
    assert bank.shape == (26, 257)
    assert np.all(bank >= 0) and np.all(bank.max(axis=1) <= 1.0)
    peaks = bank.argmax(axis=1)
    assert np.all(np.diff(peaks) > 0)  # centres increase monotonically


def test_dct_matches_scipy():
    from scipy.fft import dct

    x = np.random.default_rng(0).normal(size=(5, 26))
    np.testing.assert_allclose(dct_ii(x, 13), dct(x, type=2, norm="ortho", axis=-1)[:, :13])


def test_mfcc_shape():
    one_second = np.random.default_rng(1).normal(size=SR)
    feats = mfcc(one_second, SR)
    assert feats.shape == (99, 13)
    assert np.all(np.isfinite(feats))
    np.testing.assert_allclose(feats.mean(axis=0), 0, atol=1e-9)  # CMN


def test_mfcc_options_shape():
    x = np.random.default_rng(2).normal(size=8000)
    assert mfcc(x, 8000, n_mfcc=20, n_filters=40).shape == (99, 20)
    with pytest.raises(ValueError):
        mfcc(x, 8000, n_mfcc=30, n_filters=26)


def test_mfcc_distinguishes_tones():
    t = np.arange(SR // 2) / SR
    a = mfcc(np.sin(2 * np.pi * 300 * t), SR, cmn=False)
    b = mfcc(np.sin(2 * np.pi * 300 * t) * 0.5, SR, cmn=False)
    c = mfcc(np.sin(2 * np.pi * 2000 * t), SR, cmn=False)
    # Gain mostly changes c0; spectral shape (c1..) differs between frequencies.
    assert np.linalg.norm(a[:, 1:] - b[:, 1:]) < np.linalg.norm(a[:, 1:] - c[:, 1:]) / 10


def test_deltas_of_linear_ramp():
    feats = np.arange(20.0)[:, None] * np.ones((1, 3))
    d = deltas(feats)
    np.testing.assert_allclose(d[2:-2], 1.0)


# ------------------------------------------------------------------- DTW


@pytest.fixture
def seqs():
    rng = np.random.default_rng(3)
    return rng.normal(size=(30, 4)), rng.normal(size=(45, 4))


def test_dtw_identity(seqs):
    a, _ = seqs
    assert dtw(a, a) == pytest.approx(0.0, abs=1e-6)


def test_dtw_symmetry_and_nonnegativity(seqs):
    a, b = seqs
    assert dtw(a, b) == pytest.approx(dtw(b, a))
    assert dtw(a, b) > 0


def test_dtw_time_warp_invariance(seqs):
    a, b = seqs
    stretched = np.repeat(a, 2, axis=0)  # same sequence at half speed
    assert dtw(a, stretched) == pytest.approx(0.0, abs=1e-6)
    assert dtw(a, stretched) < dtw(a, b)


def test_dtw_matches_naive_reference(seqs):
    a, b = seqs
    n, m = len(a), len(b)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1])
            acc[i, j] = c + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    assert dtw(a, b) == pytest.approx(acc[n, m] / (n + m))


def test_dtw_band_is_upper_bound(seqs):
    a, b = seqs
    assert dtw(a, b, band=3) >= dtw(a, b) - 1e-12


def test_dtw_path(seqs):
    a, b = seqs
    dist, path = dtw(a, b, return_path=True)
    assert path[0] == (0, 0) and path[-1] == (len(a) - 1, len(b) - 1)
    steps = {(i2 - i1, j2 - j1) for (i1, j1), (i2, j2) in zip(path, path[1:], strict=False)}
    assert steps <= {(1, 0), (0, 1), (1, 1)}
    cost = sum(np.linalg.norm(a[i] - b[j]) for i, j in path)
    assert dist == pytest.approx(cost / (len(a) + len(b)))


def test_dtw_dimension_mismatch():
    with pytest.raises(ValueError):
        dtw(np.zeros((3, 2)), np.zeros((3, 3)))


# ----------------------------------------------------------- audio/synth


def test_synth_is_reproducible():
    a = synth_word("yes", rng=np.random.default_rng(7))
    b = synth_word("yes", rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(KeyError):
        synth_word("banana")


def test_trim_silence_removes_padding():
    word = synth_word("no", rng=np.random.default_rng(0), snr_db=None)
    padded = np.concatenate([np.zeros(SR), word, np.zeros(SR)])
    trimmed = trim_silence(padded, SR)
    assert len(trimmed) < len(word) + 0.2 * SR


def test_wav_roundtrip(tmp_path):
    x = 0.5 * np.sin(np.linspace(0, 100, SR))
    path = tmp_path / "x.wav"
    write_wav(path, x, SR)
    y = read_wav(path, SR)
    np.testing.assert_allclose(x, y, atol=1e-4)
    assert len(read_wav(path, 8000)) == SR // 2


def test_to_float_mono_stereo_int16():
    stereo = np.array([[32767, -32767], [16384, 16384]], dtype=np.int16)
    np.testing.assert_allclose(to_float_mono(stereo), [0.0, 16384 / 32768])


# ----------------------------------------------------------- recognition


def _trained(n_templates=1, seed=0, snr=20.0):
    rec = TemplateRecognizer()
    for label, sig in make_dataset(n_per_word=n_templates, seed=seed, snr_db=snr):
        rec.enroll(label, sig)
    return rec


def test_recognition_accuracy_synthetic():
    rec = _trained()
    test = make_dataset(n_per_word=5, seed=123, snr_db=10.0)
    acc, _ = evaluate(rec, test)
    assert acc >= 0.95, acc


def test_recognizer_reject_threshold():
    rec = _trained()
    rec.threshold = 1e-9
    assert rec.recognize(synth_word("go", rng=np.random.default_rng(5))).label is None


def test_recognizer_requires_templates():
    with pytest.raises(RuntimeError):
        TemplateRecognizer().recognize(np.zeros(SR))


def test_save_load(tmp_path):
    rec = _trained()
    path = tmp_path / "db.npz"
    rec.save(path)
    loaded = TemplateRecognizer.load(path)
    assert loaded.labels == sorted(VOCABULARY)
    sig = synth_word("left", rng=np.random.default_rng(9))
    assert loaded.recognize(sig).label == rec.recognize(sig).label == "left"


def test_cli_end_to_end(tmp_path, capsys):
    samples = tmp_path / "samples"
    db = str(tmp_path / "db.npz")
    assert (
        cli.main(["synth", "--out-dir", str(samples), "--count", "2", "--words", "up", "down"]) == 0
    )
    assert cli.main(["enroll", "--db", db, "--label", "up", str(samples / "up_0.wav")]) == 0
    assert cli.main(["enroll", "--db", db, "--label", "down", str(samples / "down_0.wav")]) == 0
    capsys.readouterr()
    args = ["recognize", "--db", db, str(samples / "up_1.wav"), str(samples / "down_1.wav")]
    assert cli.main(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert ": up (" in out[0] and ": down (" in out[1]
