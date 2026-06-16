# tools/test_birdnet_direct.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np

import birdnet_analyzer.config as cfg
from birdnet_analyzer import model


def sigmoid(x: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """
    BirdNET-Analyzer の感度補正に近い sigmoid。
    まずは sensitivity=1.0 で使う。
    """
    return 1.0 / (1.0 + np.exp(-sensitivity * x))


def load_labels(path: str | Path) -> list[str]:
    path = Path(path)
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)

    return labels


def split_label(label: str) -> tuple[str, str]:
    if "_" in label:
        sci, common = label.split("_", 1)
        return sci, common
    return label, label


def load_audio_3sec(path: str | Path, sr: int = 48000, seconds: float = 3.0) -> np.ndarray:
    y, _ = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        duration=seconds,
        res_type="kaiser_fast",
    )

    n = int(sr * seconds)

    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]

    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min_conf", type=float, default=0.01)
    args = parser.parse_args()

    # BirdNETの基本設定
    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
    cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SIG_OVERLAP = 0.0
    cfg.SIG_MINLEN = 1.0
    cfg.TFLITE_THREADS = 1
    cfg.CUSTOM_CLASSIFIER = None
    cfg.USE_PERCH = False

    labels = load_labels(cfg.LABELS_FILE)
    cfg.LABELS = labels

    print("MODEL:", cfg.MODEL_PATH)
    print("LABELS:", cfg.LABELS_FILE)
    print("N_LABELS:", len(labels))

    t0 = time.monotonic()
    model.load_model()
    t1 = time.monotonic()
    print(f"load_model elapsed: {t1 - t0:.3f}s")

    y = load_audio_3sec(args.wav, sr=cfg.SAMPLE_RATE, seconds=cfg.SIG_LENGTH)

    # warmup
    _ = model.predict([y])

    t2 = time.monotonic()
    pred = model.predict([y])
    t3 = time.monotonic()

    pred = np.asarray(pred)

    print("raw pred shape:", pred.shape)
    print(f"predict elapsed: {t3 - t2:.3f}s")

    scores = pred[0]

    # 出力が logits っぽい場合に sigmoid。
    # もしすでに 0-1 なら、大きくは壊れないように範囲で判断する。
    if float(np.min(scores)) < 0.0 or float(np.max(scores)) > 1.0:
        scores = sigmoid(scores)

    idxs = np.argsort(scores)[::-1]

    print()
    print("Top predictions:")
    for idx in idxs[:args.top]:
        conf = float(scores[idx])
        if conf < args.min_conf:
            continue

        label = labels[idx]
        sci, common = split_label(label)

        print(f"{conf:.4f}\t{sci}\t{common}")


if __name__ == "__main__":
    main()