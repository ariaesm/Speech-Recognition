# speech-recognition — offline voice commands with MFCC + DTW

A small, dependency-light **offline keyword recognizer** for isolated voice commands
("yes", "stop", "left", …). Features are computed with **MFCCs implemented from scratch in
NumPy**, and words are recognised by **dynamic time warping (DTW)** against templates you
record yourself. No cloud API, no internet connection, no neural network.

> Looking for cloud transcription of free speech instead? See the sibling project
> [python-program-to-convert-speech-to-text](https://github.com/ariaesm/python-program-to-convert-speech-to-text),
> which uses the Google Speech API. This repository is the opposite trade-off: tiny
> vocabulary, fully local, every step readable.

## Why

- Learn how classic speech front-ends work: each MFCC stage is a short, tested function.
- Recognise a handful of commands on a device without network access.
- Enroll new words in seconds from one or two WAV recordings — no training run.

## Features

- **MFCC pipeline** (`kws/mfcc.py`): pre-emphasis → framing (25 ms / 10 ms) → Hamming window
  → FFT power spectrum → 26-band triangular mel filterbank → log → DCT-II → liftering →
  cepstral mean normalisation, plus regression deltas.
- **DTW** (`kws/dtw.py`): Euclidean frame cost, steps (1,0)/(0,1)/(1,1), length-normalised
  distance, optional Sakoe–Chiba band, optional warping path; vectorised along anti-diagonals.
- **Recognizer** (`kws/recognizer.py`): energy-based silence trimming, nearest-template
  classification, optional rejection threshold, templates saved as `.npz`.
- **Synthetic words** (`kws/synth.py`): 8 tone-pattern "words" with random speaking rate
  (±25 %), pitch (±4 %), harmonics, amplitude, silence padding and white noise at a chosen SNR,
  so the demo and tests run without any recordings.
- **CLI**: `synth`, `enroll`, `recognize`, `demo`.

## Install

```bash
git clone https://github.com/ariaesm/Speech-Recognition.git
cd Speech-Recognition
python3 -m pip install -r requirements.txt   # numpy, scipy
```

## Usage

### Demo on synthetic data

```text
$ python3 -m kws demo
vocabulary: yes, no, up, down, left, right, stop, go
templates per word: 2, test utterances per word: 20, SNR: 20.0 dB
accuracy: 100.0% (160/160)
recognition time: 19.6 ms per utterance
```

Accuracy measured with `python3 -m kws demo --templates N --snr S` (160 test utterances,
seed 0). These are results on the synthetic tone-pattern words, not on human speech:

| Templates per word | SNR | Accuracy |
| --- | --- | --- |
| 2 | 20 dB | 100.0 % (160/160) |
| 1 | 0 dB | 100.0 % (160/160) |
| 1 | −5 dB | 98.8 % (158/160) |
| 1 | −10 dB | 79.4 % (127/160) |
| 3 | −10 dB | 84.4 % (135/160) |

Timing depends on the machine; the figure above is from one run on the development machine.

### Enroll and recognise WAV files

Any mono or stereo WAV works (it is converted to 16 kHz mono). To try it without a
microphone, generate synthetic samples first:

```text
$ python3 -m kws synth --out-dir samples --count 3
wrote 24 files to samples

$ python3 -m kws enroll --label yes samples/yes_0.wav
enrolled 1 template(s) as 'yes'; templates.npz has 1 template(s) for ['yes']
$ python3 -m kws enroll --label no samples/no_0.wav
$ python3 -m kws enroll --label up samples/up_0.wav
$ python3 -m kws enroll --label down samples/down_0.wav
enrolled 1 template(s) as 'down'; templates.npz has 4 template(s) for ['down', 'no', 'up', 'yes']

$ python3 -m kws recognize samples/yes_2.wav samples/down_1.wav samples/up_2.wav
samples/yes_2.wav: yes (distance 13.622)
samples/down_1.wav: down (distance 12.081)
samples/up_2.wav: up (distance 15.060)

$ python3 -m kws recognize -v samples/no_1.wav
samples/no_1.wav: no (distance 11.900)
          no 11.900
         yes 32.321
          up 33.381
        down 35.586
```

Words that were never enrolled can be rejected with a distance threshold:

```text
$ python3 -m kws recognize --threshold 20 samples/yes_1.wav samples/left_1.wav samples/stop_1.wav
samples/yes_1.wav: yes (distance 12.614)
samples/left_1.wav: <rejected> (distance 28.487)
samples/stop_1.wav: <rejected> (distance 34.856)
```

With your own voice: record each command 2–3 times (quiet room, ~1 s each), enroll every file
under its label, and pick a threshold from the `-v` scores.

### Python API

```python
from kws import TemplateRecognizer
from kws.audio import read_wav

rec = TemplateRecognizer()
rec.enroll("stop", read_wav("stop_1.wav"))
rec.enroll("go", read_wav("go_1.wav"))
print(rec.recognize(read_wav("unknown.wav")).label)
```

## Tests

```bash
python3 -m pip install pytest
python3 -m pytest
```

31 tests cover framing, window, mel conversion and filterbank, DCT (checked against
`scipy.fft.dct`), MFCC shapes, deltas, DTW properties (identity, symmetry, time-warp
invariance, band upper bound, path validity, agreement with a naive reference
implementation), WAV I/O, silence trimming, recognition accuracy on synthetic data,
template persistence and the CLI. CI runs `ruff` and `pytest` on Python 3.12.

## Limitations

- Template matching is speaker-dependent and suited to small vocabularies (tens of words).
- Isolated words only: no continuous speech and no streaming wake-word detection.
- Synthetic results above say nothing about accuracy on real recordings; measure on your own.

## License

[Apache License 2.0](LICENSE)
