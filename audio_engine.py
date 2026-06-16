# audio_engine.py
from __future__ import annotations

import threading
from collections import deque

import numpy as np
import sounddevice as sd

from state import update_state


class AudioEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.sample_rate = int(cfg["audio"]["sample_rate"])
        self.channels = int(cfg["audio"]["channels"])
        self.block_size = int(cfg["audio"]["block_size"])

        self.keep_seconds = 30.0
        self.max_samples = int(self.sample_rate * self.keep_seconds)

        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
        self.stream = None

    def find_audio_device(self) -> int | None:
        device_index = self.cfg["audio"].get("device_index", None)
        if device_index is not None:
            return int(device_index)

        name_contains = self.cfg["audio"].get("device_name_contains") or None
        if not name_contains:
            return None

        name_contains = name_contains.lower()
        devices = sd.query_devices()

        for i, d in enumerate(devices):
            name = str(d.get("name", "")).lower()
            max_in = int(d.get("max_input_channels", 0))

            if name_contains in name and max_in >= self.channels:
                return i

        raise RuntimeError(
            f"Audio input device containing '{name_contains}' was not found."
        )

    def callback(self, indata, frames, time_info, status):
        if status:
            update_state("audio", {"status": f"audio warning: {status}"})

        data = np.asarray(indata, dtype=np.float32)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        ch1 = data[:, 0]
        ch2 = data[:, 1] if data.shape[1] > 1 else data[:, 0]

        level_ch1 = float(np.sqrt(np.mean(ch1 ** 2)))
        level_ch2 = float(np.sqrt(np.mean(ch2 ** 2)))

        update_state("audio", {
            "level_ch1": level_ch1,
            "level_ch2": level_ch2,
        })

        with self.lock:
            for row in data:
                self.buffer.append(row.copy())

    def start(self) -> None:
        device = self.find_audio_device()

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="float32",
            device=device,
            callback=self.callback,
        )

        self.stream.start()

        update_state("audio", {
            "status": f"audio running: device={device}",
        })

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        update_state("audio", {
            "status": "audio stopped",
        })

    def get_recent_audio(self, seconds: float, channel: int) -> np.ndarray:
        n = int(seconds * self.sample_rate)

        with self.lock:
            arr = np.array(self.buffer, dtype=np.float32)

        if arr.size == 0:
            return np.zeros(n, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.shape[0] < n:
            pad = np.zeros((n - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        else:
            arr = arr[-n:]

        if channel >= arr.shape[1]:
            channel = 0

        y = arr[:, channel].astype(np.float32)

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.99:
            y = y / peak * 0.95

        return y