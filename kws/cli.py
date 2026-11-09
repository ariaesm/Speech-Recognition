"""Command-line interface: synth, enroll, recognize, demo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .audio import read_wav, write_wav
from .recognizer import TemplateRecognizer, evaluate
from .synth import VOCABULARY, make_dataset, synth_word


def cmd_synth(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    words = args.words or list(VOCABULARY)
    for word in words:
        for i in range(args.count):
            path = out / f"{word}_{i}.wav"
            write_wav(path, synth_word(word, args.rate, rng, snr_db=args.snr))
    print(f"wrote {len(words) * args.count} files to {out}")
    return 0


def cmd_enroll(args) -> int:
    db = Path(args.db)
    rec = TemplateRecognizer.load(db) if db.exists() else TemplateRecognizer(args.rate)
    for wav in args.wavs:
        rec.enroll(args.label, read_wav(wav, rec.sample_rate))
    rec.save(db)
    print(
        f"enrolled {len(args.wavs)} template(s) as {args.label!r}; {db} has "
        f"{len(rec.templates)} template(s) for {rec.labels}"
    )
    return 0


def cmd_recognize(args) -> int:
    rec = TemplateRecognizer.load(args.db, band=args.band, threshold=args.threshold)
    for wav in args.wavs:
        result = rec.recognize(read_wav(wav, rec.sample_rate))
        label = result.label if result.label is not None else "<rejected>"
        print(f"{wav}: {label} (distance {result.distance:.3f})")
        if args.verbose:
            for name, score in sorted(result.scores.items(), key=lambda kv: kv[1]):
                print(f"    {name:>8s} {score:.3f}")
    return 0


def cmd_demo(args) -> int:
    words = list(VOCABULARY)
    train = make_dataset(words, args.templates, seed=args.seed, snr_db=args.snr)
    test = make_dataset(words, args.tests, seed=args.seed + 1, snr_db=args.snr)
    rec = TemplateRecognizer(band=args.band)
    for label, signal in train:
        rec.enroll(label, signal)
    t0 = time.perf_counter()
    acc, confusion = evaluate(rec, test)
    elapsed = time.perf_counter() - t0
    print(f"vocabulary: {', '.join(words)}")
    print(
        f"templates per word: {args.templates}, test utterances per word: {args.tests}, "
        f"SNR: {args.snr} dB"
    )
    print(f"accuracy: {acc:.1%} ({round(acc * len(test))}/{len(test)})")
    print(f"recognition time: {1000 * elapsed / len(test):.1f} ms per utterance")
    errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
    for (true, pred), n in sorted(errors.items()):
        print(f"  confused {true} -> {pred}: {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kws", description="Offline MFCC + DTW voice-command recognizer"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write synthetic word WAV files")
    s.add_argument("--out-dir", default="samples")
    s.add_argument("--count", type=int, default=3, help="utterances per word")
    s.add_argument("--words", nargs="*", choices=sorted(VOCABULARY))
    s.add_argument("--snr", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rate", type=int, default=16000)
    s.set_defaults(func=cmd_synth)

    e = sub.add_parser("enroll", help="add WAV templates for a label")
    e.add_argument("--db", default="templates.npz")
    e.add_argument("--label", required=True)
    e.add_argument("--rate", type=int, default=16000)
    e.add_argument("wavs", nargs="+")
    e.set_defaults(func=cmd_enroll)

    r = sub.add_parser("recognize", help="recognize WAV files against enrolled templates")
    r.add_argument("--db", default="templates.npz")
    r.add_argument("--band", type=int, default=None, help="Sakoe-Chiba band in frames")
    r.add_argument("--threshold", type=float, default=None, help="reject above this distance")
    r.add_argument("-v", "--verbose", action="store_true", help="print all label scores")
    r.add_argument("wavs", nargs="+")
    r.set_defaults(func=cmd_recognize)

    d = sub.add_parser("demo", help="evaluate on synthetic data (no files needed)")
    d.add_argument("--templates", type=int, default=2)
    d.add_argument("--tests", type=int, default=20)
    d.add_argument("--snr", type=float, default=20.0)
    d.add_argument("--band", type=int, default=None)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
