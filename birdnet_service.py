# birdnet_service.py
from __future__ import annotations

import csv
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import birdnet_analyzer.config as bcfg
from birdnet_analyzer import model

from state import update_state


class LatestOnlyQueue:
    """
    解析待ちを最新1件だけにするキュー。

    CLI版ではBirdNET解析が遅い場合に古いWAVをため込まないために使う。
    """

    def __init__(self):
        self.q = queue.Queue(maxsize=1)

    def put_latest(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(item)

    def get(self, timeout: float | None = None):
        return self.q.get(timeout=timeout)


class BirdNETService:
    def __init__(self, cfg: dict, audio_engine):
        self.cfg = cfg
        self.audio_engine = audio_engine

        self.inference_mode = str(
            self.cfg["birdnet"].get("inference_mode", "direct")
        ).lower()

        if self.inference_mode not in {"direct", "cli"}:
            raise ValueError(
                "birdnet.inference_mode must be either 'direct' or 'cli'"
            )

        self.thread = None
        self.capture_thread = None
        self.analyze_thread = None
        self.stop_event = threading.Event()
        self.wav_queue = LatestOnlyQueue()

        self.labels: list[str] = []
        self.ja_name_map: dict[str, str] = {}
        self.location_filter: np.ndarray | None = None
        self.last_elapsed_seconds = 0.0

        self.ja_name_map = self.load_birdnet_ja_labels()

        if self.inference_mode == "direct":
            self.setup_birdnet_direct()

        update_state("birdnet", {
            "enabled": True,
            "running": False,
            "status": "ready",
            "capture_status": "ready",
            "analyze_status": f"ready: inference_mode={self.inference_mode}",
            "last_elapsed_seconds": None,
            "latest_detection": None,
        })

    # ---------------------------------------------------------------------
    # 共通処理
    # ---------------------------------------------------------------------

    def start(self) -> None:
        if self.inference_mode == "direct":
            self.start_direct()
        else:
            self.start_cli()

    def stop(self) -> None:
        self.stop_event.set()

        update_state("birdnet", {
            "enabled": True,
            "running": False,
            "status": "stopping",
            "capture_status": "stopping",
            "analyze_status": "stopping",
        })

    def append_csv(self, path: str | Path, row: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        new_file = not path.exists()

        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def load_birdnet_ja_labels(self) -> dict[str, str]:
        """
        BirdNET_GLOBAL_6K_V2.4_Labels_ja.txt を読み、
        Scientific name -> Japanese name の辞書を作る。
        """
        path = self.cfg["birdnet"].get("labels_ja_path")

        if not path:
            return {}

        path = Path(path)

        if not path.exists():
            update_state("birdnet", {
                "analyze_status": f"Japanese label file not found: {path}",
            })
            return {}

        mapping: dict[str, str] = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or "_" not in line:
                    continue

                sci, ja = line.split("_", 1)
                sci = sci.strip()
                ja = ja.strip()

                if sci and ja:
                    mapping[sci] = ja

        return mapping

    def load_labels(self, path: str | Path) -> list[str]:
        path = Path(path)
        labels: list[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)

        return labels

    def split_label(self, label: str) -> tuple[str, str]:
        if "_" in label:
            sci, common = label.split("_", 1)
            return sci, common
        return label, label

    def sigmoid(self, x: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-sensitivity * x))

    def normalize_audio_length(self, y: np.ndarray, seconds: float, sr: int) -> np.ndarray:
        """
        BirdNET入力用に、指定秒数ぴったりのfloat32 mono配列にする。
        長い場合は末尾、つまり最新側を使う。
        """
        target_len = int(seconds * sr)

        y = np.asarray(y, dtype=np.float32).flatten()

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[-target_len:]

        return y.astype(np.float32)

    def add_japanese_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BirdNET結果DataFrameの Scientific name から和名を付ける。
        和名が見つからない場合は Common name、
        さらに無ければ Scientific name を表示名にする。
        """
        if df.empty:
            return df

        df = df.copy()

        if "Scientific name" not in df.columns:
            df["Japanese name"] = ""
            if "Common name" in df.columns:
                df["Display name"] = df["Common name"]
            else:
                df["Display name"] = ""
            return df

        df["Japanese name"] = df["Scientific name"].map(self.ja_name_map)

        if "Common name" in df.columns:
            df["Display name"] = (
                df["Japanese name"]
                .fillna(df["Common name"])
                .fillna(df["Scientific name"])
            )
        else:
            df["Display name"] = (
                df["Japanese name"]
                .fillna(df["Scientific name"])
            )

        return df

    def filter_display_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        UI表示対象の候補に絞る。

        allowed_scientific_names が空なら学名フィルタなし。
        allowed_japanese_names が空なら和名フィルタなし。
        display_confidence が指定されていれば、それ未満を非表示。
        """
        if df.empty:
            return df

        df = self.add_japanese_names(df)

        bird_cfg = self.cfg["birdnet"]
        allowed_sci = bird_cfg.get("allowed_scientific_names") or []
        allowed_ja = bird_cfg.get("allowed_japanese_names") or []
        display_conf = float(
            bird_cfg.get("display_confidence", bird_cfg.get("min_confidence", 0.01))
        )

        if "Confidence" in df.columns:
            df = df[df["Confidence"] >= display_conf].copy()

        if allowed_sci and "Scientific name" in df.columns:
            df = df[df["Scientific name"].isin(allowed_sci)].copy()

        if allowed_ja and "Japanese name" in df.columns:
            df = df[df["Japanese name"].isin(allowed_ja)].copy()

        if "Confidence" in df.columns:
            df = df.sort_values("Confidence", ascending=False)

        return df

    def dataframe_to_top(self, df: pd.DataFrame, max_items: int = 3) -> list[dict]:
        top = []

        if df.empty:
            return top

        for _, r in df.head(max_items).iterrows():
            confidence = float(r.get("Confidence", 0.0))
            sci = str(r.get("Scientific name", ""))
            common = str(r.get("Common name", ""))

            ja_value = r.get("Japanese name", "")
            if pd.isna(ja_value):
                ja = ""
            else:
                ja = str(ja_value)

            display_value = r.get("Display name", "")
            if pd.isna(display_value):
                display = common or sci
            else:
                display = str(display_value)

            top.append({
                "display_name": display,
                "japanese_name": ja,
                "common_name": common,
                "scientific_name": sci,
                "confidence": round(confidence, 4),
            })

        return top

    def publish_result(
        self,
        top: list[dict],
        raw_count: int,
        display_count: int,
        elapsed: float,
        source: str,
        wav_path: str = "",
        queued_at: str = "",
    ) -> None:
        """
        UI状態とログCSVを更新する。
        direct版・cli版共通。
        """
        elapsed_rounded = round(float(elapsed), 3)
        self.last_elapsed_seconds = elapsed_rounded

        update_state("birdnet", {
            "last_elapsed_seconds": elapsed_rounded,
        })

        if top:
            latest = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": "判定しました",
                "top": top,
            }

            update_state("birdnet", {
                "status": "detected",
                "analyze_status": (
                    f"detected[{source}]: raw={raw_count}, "
                    f"display={display_count}, elapsed={elapsed_rounded:.3f}s"
                ),
                "latest_detection": latest,
            })

            self.append_csv(self.cfg["birdnet"]["log_csv"], {
                "datetime": datetime.now().isoformat(timespec="seconds"),
                "mode": source,
                "rank1_display_name": top[0]["display_name"],
                "rank1_japanese_name": top[0]["japanese_name"],
                "rank1_common_name": top[0]["common_name"],
                "rank1_scientific_name": top[0]["scientific_name"],
                "rank1_confidence": top[0]["confidence"],
                "raw_count": raw_count,
                "display_count": display_count,
                "elapsed_seconds": elapsed_rounded,
                "queued_at": queued_at,
                "wav": wav_path,
            })

        else:
            update_state("birdnet", {
                "status": "no candidate",
                "analyze_status": (
                    f"no candidate[{source}]: raw={raw_count}, "
                    f"display={display_count}, elapsed={elapsed_rounded:.3f}s"
                ),
                "latest_detection": {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "message": "候補なし",
                    "top": [],
                },
            })

    # ---------------------------------------------------------------------
    # direct版
    # ---------------------------------------------------------------------

    def setup_birdnet_direct(self) -> None:
        """
        BirdNETモデル・英語ラベル・地域季節フィルタを初期化する。
        subprocessではなく、同じPythonプロセス内でモデルを常駐させる。
        """
        bird_cfg = self.cfg["birdnet"]

        bcfg.MODEL_PATH = bcfg.BIRDNET_MODEL_PATH
        bcfg.LABELS_FILE = bcfg.BIRDNET_LABELS_FILE
        bcfg.SAMPLE_RATE = bcfg.BIRDNET_SAMPLE_RATE
        bcfg.SIG_LENGTH = bcfg.BIRDNET_SIG_LENGTH
        bcfg.SIG_OVERLAP = 0.0
        bcfg.SIG_MINLEN = 1.0
        bcfg.TFLITE_THREADS = int(bird_cfg.get("tflite_threads", 1))
        bcfg.CUSTOM_CLASSIFIER = None
        bcfg.USE_PERCH = False
        bcfg.APPLY_SIGMOID = True
        bcfg.SIGMOID_SENSITIVITY = 1.0

        self.labels = self.load_labels(bcfg.LABELS_FILE)
        bcfg.LABELS = self.labels

        t0 = time.monotonic()
        model.load_model()
        t1 = time.monotonic()

        self.location_filter = None
        if bool(bird_cfg.get("use_location_filter", False)):
            lat = float(bird_cfg.get("latitude", -1))
            lon = float(bird_cfg.get("longitude", -1))
            week = int(bird_cfg.get("week", -1))

            if lat != -1 and lon != -1 and week != -1:
                try:
                    self.location_filter = np.asarray(
                        model.predict_filter(lat, lon, week),
                        dtype=np.float32,
                    )
                except Exception as e:
                    self.location_filter = None
                    update_state("birdnet", {
                        "analyze_status": f"location filter load error: {e}",
                    })

        update_state("birdnet", {
            "status": "ready",
            "capture_status": "ready",
            "analyze_status": (
                f"BirdNET direct model loaded: labels={len(self.labels)}, "
                f"load_elapsed={t1 - t0:.3f}s"
            ),
            "last_elapsed_seconds": None,
        })

    def start_direct(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            update_state("birdnet", {
                "enabled": True,
                "running": True,
                "status": "already running",
            })
            return

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.direct_loop,
            daemon=True,
        )
        self.thread.start()

        update_state("birdnet", {
            "enabled": True,
            "running": True,
            "status": "running",
            "capture_status": "direct capture running",
            "analyze_status": "direct analyze running",
            "last_elapsed_seconds": None,
            "latest_detection": None,
        })

    def run_birdnet_direct(self, y: np.ndarray) -> pd.DataFrame:
        """
        BirdNETモデルを直接呼び出してDataFrameを返す。
        WAV/CSVファイルは作らない。
        """
        bird_cfg = self.cfg["birdnet"]
        min_conf = float(bird_cfg.get("min_confidence", 0.01))

        started = time.monotonic()

        pred = model.predict([y])
        elapsed = time.monotonic() - started

        scores = np.asarray(pred)[0].astype(np.float32)

        # 出力がlogits風の場合のみsigmoid補正。
        # BirdNET v2.4 TFLiteでは通常0-1のスコアが返る。
        if float(np.min(scores)) < 0.0 or float(np.max(scores)) > 1.0:
            scores = self.sigmoid(scores)

        if self.location_filter is not None and len(self.location_filter) == len(scores):
            scores = scores * self.location_filter

        rows = []
        idxs = np.argsort(scores)[::-1]

        for idx in idxs:
            confidence = float(scores[idx])
            if confidence < min_conf:
                break

            label = self.labels[idx]
            sci, common = self.split_label(label)
            ja = self.ja_name_map.get(sci, "")
            display = ja or common or sci

            rows.append({
                "Start (s)": 0.0,
                "End (s)": float(bird_cfg.get("window_seconds", 3.0)),
                "Scientific name": sci,
                "Common name": common,
                "Japanese name": ja,
                "Display name": display,
                "Confidence": confidence,
            })

        self.last_elapsed_seconds = round(elapsed, 3)

        update_state("birdnet", {
            "last_elapsed_seconds": self.last_elapsed_seconds,
        })

        return pd.DataFrame(rows)

    def direct_loop(self) -> None:
        cfg = self.cfg

        ch = int(cfg["audio"]["birdnet_channel"])
        interval = float(cfg["birdnet"]["analyze_interval_seconds"])
        window_sec = float(cfg["birdnet"]["window_seconds"])

        update_state("birdnet", {
            "enabled": True,
            "running": True,
            "status": "running",
            "capture_status": "direct capture running",
            "analyze_status": "direct analyze running",
        })

        next_time = time.monotonic()

        while not self.stop_event.is_set():
            try:
                now = time.monotonic()
                if now < next_time:
                    time.sleep(min(0.02, next_time - now))
                    continue

                next_time += interval

                y = self.audio_engine.get_recent_audio(window_sec, ch)
                y = self.normalize_audio_length(
                    y,
                    seconds=bcfg.SIG_LENGTH,
                    sr=bcfg.SAMPLE_RATE,
                )

                rms = float(np.sqrt(np.mean(y ** 2)))
                peak = float(np.max(np.abs(y))) if y.size else 0.0

                if rms < 0.002:
                    update_state("birdnet", {
                        "status": "waiting",
                        "capture_status": (
                            f"direct waiting for sound: "
                            f"rms={rms:.5f}, peak={peak:.5f}"
                        ),
                        "latest_detection": {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "message": "音を待っています",
                            "top": [],
                        },
                    })
                    continue

                update_state("birdnet", {
                    "capture_status": (
                        f"direct captured buffer: "
                        f"rms={rms:.5f}, peak={peak:.5f}"
                    ),
                    "analyze_status": "direct analyzing model",
                })

                df_raw = self.run_birdnet_direct(y)
                raw_count = len(df_raw)

                df = self.filter_display_results(df_raw)
                display_count = len(df)

                top = self.dataframe_to_top(df, max_items=3)

                self.publish_result(
                    top=top,
                    raw_count=raw_count,
                    display_count=display_count,
                    elapsed=self.last_elapsed_seconds,
                    source="direct",
                    wav_path="",
                    queued_at="",
                )

            except Exception as e:
                update_state("birdnet", {
                    "status": "error",
                    "analyze_status": f"direct loop error: {e}",
                })
                time.sleep(0.5)

        update_state("birdnet", {
            "enabled": True,
            "running": False,
            "status": "stopped",
            "capture_status": "direct stopped",
            "analyze_status": "direct stopped",
        })

    # ---------------------------------------------------------------------
    # CLI版
    # ---------------------------------------------------------------------

    def start_cli(self) -> None:
        if (
            self.capture_thread is not None
            and self.capture_thread.is_alive()
            and self.analyze_thread is not None
            and self.analyze_thread.is_alive()
        ):
            update_state("birdnet", {
                "enabled": True,
                "running": True,
                "status": "already running",
            })
            return

        self.stop_event.clear()

        self.capture_thread = threading.Thread(
            target=self.cli_capture_loop,
            daemon=True,
        )
        self.analyze_thread = threading.Thread(
            target=self.cli_analyze_loop,
            daemon=True,
        )

        self.capture_thread.start()
        self.analyze_thread.start()

        update_state("birdnet", {
            "enabled": True,
            "running": True,
            "status": "running",
            "capture_status": "cli capture running",
            "analyze_status": "cli analyze running",
            "last_elapsed_seconds": None,
            "latest_detection": None,
        })

    def cli_capture_loop(self) -> None:
        """
        直近 window_seconds 秒を、analyze_interval_seconds ごとにWAV化する。
        CLI解析が詰まっていても、最新1件だけキューに残す。
        """
        cfg = self.cfg

        sr = int(cfg["audio"]["sample_rate"])
        ch = int(cfg["audio"]["birdnet_channel"])
        window_sec = float(cfg["birdnet"]["window_seconds"])
        interval = float(cfg["birdnet"]["analyze_interval_seconds"])

        input_dir = Path(cfg["birdnet"]["input_dir"])
        input_dir.mkdir(parents=True, exist_ok=True)

        next_time = time.monotonic()

        while not self.stop_event.is_set():
            try:
                now = time.monotonic()
                if now < next_time:
                    time.sleep(min(0.05, next_time - now))
                    continue

                next_time += interval

                y = self.audio_engine.get_recent_audio(window_sec, ch)

                rms = float(np.sqrt(np.mean(y ** 2)))
                peak = float(np.max(np.abs(y))) if y.size else 0.0

                if rms < 0.002:
                    update_state("birdnet", {
                        "status": "waiting",
                        "capture_status": (
                            f"cli waiting for sound: "
                            f"rms={rms:.5f}, peak={peak:.5f}"
                        ),
                        "latest_detection": {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "message": "音を待っています",
                            "top": [],
                        },
                    })
                    continue

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                wav_path = input_dir / f"birdnet_{ts}.wav"

                sf.write(wav_path, y, sr, subtype="PCM_16")

                self.wav_queue.put_latest({
                    "wav_path": wav_path,
                    "rms": rms,
                    "peak": peak,
                    "created_at": datetime.now().isoformat(timespec="milliseconds"),
                })

                update_state("birdnet", {
                    "capture_status": (
                        f"cli queued latest wav: {wav_path.name}, "
                        f"rms={rms:.5f}, peak={peak:.5f}"
                    ),
                })

            except Exception as e:
                update_state("birdnet", {
                    "status": "error",
                    "capture_status": f"cli capture loop error: {e}",
                })
                time.sleep(0.5)

        update_state("birdnet", {
            "capture_status": "cli capture stopped",
        })

    def cli_analyze_loop(self) -> None:
        """
        最新WAVだけを順次BirdNET CLIで解析する。
        """
        while not self.stop_event.is_set():
            try:
                item = self.wav_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            wav_path = Path(item["wav_path"])
            queued_at = item.get("created_at", "")

            started = time.monotonic()

            update_state("birdnet", {
                "analyze_status": f"cli analyzing: {wav_path.name}",
            })

            df_raw = self.run_birdnet_cli(wav_path)
            raw_count = len(df_raw)

            df = self.filter_display_results(df_raw)
            display_count = len(df)

            elapsed = time.monotonic() - started
            self.last_elapsed_seconds = round(elapsed, 3)

            top = self.dataframe_to_top(df, max_items=3)

            self.publish_result(
                top=top,
                raw_count=raw_count,
                display_count=display_count,
                elapsed=self.last_elapsed_seconds,
                source="cli",
                wav_path=str(wav_path),
                queued_at=queued_at,
            )

        update_state("birdnet", {
            "enabled": True,
            "running": False,
            "status": "stopped",
            "analyze_status": "cli analyze stopped",
        })

    def run_birdnet_cli(self, wav_path: Path) -> pd.DataFrame:
        cfg = self.cfg

        out_dir = Path(cfg["birdnet"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "birdnet_analyzer.analyze",
            str(wav_path),
            "-o",
            str(out_dir),
            "--rtype",
            "csv",
            "--min_conf",
            str(cfg["birdnet"]["min_confidence"]),
            "--lat",
            str(cfg["birdnet"]["latitude"]),
            "--lon",
            str(cfg["birdnet"]["longitude"]),
            "--week",
            str(cfg["birdnet"]["week"]),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )

        if result.returncode != 0:
            update_state("birdnet", {
                "status": "error",
                "analyze_status": "BirdNET CLI error: " + result.stderr[-500:],
            })
            return pd.DataFrame()

        csvs = sorted(out_dir.glob(f"{wav_path.stem}*.csv"))

        if not csvs:
            update_state("birdnet", {
                "analyze_status": "BirdNET CLI finished but no CSV was found",
            })
            return pd.DataFrame()

        try:
            df = pd.read_csv(csvs[-1])
            df = self.add_japanese_names(df)
            return df

        except Exception as e:
            update_state("birdnet", {
                "status": "error",
                "analyze_status": f"CSV read error: {e}",
            })
            return pd.DataFrame()