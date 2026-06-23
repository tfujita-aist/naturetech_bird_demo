# birdnet_service.py
from __future__ import annotations

import csv
import inspect
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from state import update_state


def clamp01(x: float | None) -> float | None:
    if x is None:
        return None
    if not math.isfinite(float(x)):
        return None
    return float(max(0.0, min(1.0, x)))


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def normalize_label_key(value: str | None) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("　", " ")
        .replace("_", " ")
        .replace("-", " ")
    )


class BirdNETService:
    def __init__(self, cfg: dict, audio_engine):
        self.cfg = cfg
        self.audio_engine = audio_engine

        self.birdnet_cfg = cfg.get("birdnet", {})
        self.audio_cfg = cfg.get("audio", {})

        self.sample_rate = int(self.audio_cfg.get("sample_rate", 48000))
        self.birdnet_channel = int(self.audio_cfg.get("birdnet_channel", 0))

        self.window_seconds = float(self.birdnet_cfg.get("window_seconds", 3.0))
        self.interval_seconds = float(self.birdnet_cfg.get("analyze_interval_seconds", 1.0))
        self.min_confidence = float(self.birdnet_cfg.get("min_confidence", 0.01))
        self.display_confidence = float(self.birdnet_cfg.get("display_confidence", 0.10))
        self.tflite_threads = int(self.birdnet_cfg.get("tflite_threads", 1))

        self.allowed_scientific_names = set(
            normalize_label_key(x)
            for x in self.birdnet_cfg.get("allowed_scientific_names", [])
            if str(x).strip()
        )
        self.allowed_japanese_names = set(
            normalize_label_key(x)
            for x in self.birdnet_cfg.get("allowed_japanese_names", [])
            if str(x).strip()
        )

        self.running = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_input_samples = int(self.window_seconds * 48000)
        self.model_sample_rate = 48000

        self.labels: list[dict] = []
        self.geo_scores: dict[str, float] = {}
        self.geo_enabled = bool(self.birdnet_cfg.get("use_location_filter", False))
        self.geo_status = "not used"

        self._load_audio_model()
        self._load_labels()
        self._load_geo_scores()

    def _load_audio_model(self) -> None:
        model_path = Path(self.birdnet_cfg.get("model_path", "")).expanduser()

        if not model_path.exists():
            raise FileNotFoundError(f"BirdNET audio model was not found: {model_path}")

        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter
            except Exception as e:
                raise RuntimeError(
                    "TensorFlow Lite interpreter was not available. "
                    "Install tensorflow or tflite-runtime."
                ) from e

        self.interpreter = Interpreter(
            model_path=str(model_path),
            num_threads=self.tflite_threads,
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        if len(input_shape) >= 2:
            self.model_input_samples = int(input_shape[-1])
        else:
            self.model_input_samples = int(self.window_seconds * self.model_sample_rate)

        print(
            "BirdNET direct model loaded: "
            f"model_file={model_path.name}, input_samples={self.model_input_samples}",
            flush=True,
        )

    def _find_package_labels_file(self) -> Path | None:
        try:
            import birdnet_analyzer

            root = Path(birdnet_analyzer.__file__).resolve().parent
            candidates = list(root.rglob("BirdNET_GLOBAL_6K_V2.4_Labels_ja.txt"))
            if candidates:
                return candidates[0]
        except Exception:
            pass

        return None

    def _load_labels(self) -> None:
        labels_path_raw = self.birdnet_cfg.get("labels_ja_path", "")
        labels_path = Path(labels_path_raw).expanduser() if labels_path_raw else None

        if labels_path is None or not labels_path.exists():
            fallback = self._find_package_labels_file()
            if fallback is not None:
                labels_path = fallback

        if labels_path is None or not labels_path.exists():
            raise FileNotFoundError(
                "BirdNET labels file was not found. "
                "Set birdnet.labels_ja_path in config.yaml."
            )

        labels: list[dict] = []

        with open(labels_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                raw = line.strip()
                if not raw:
                    continue

                scientific_name = raw
                common_name = raw
                japanese_name = raw

                parts = raw.split("_")
                if len(parts) >= 3:
                    scientific_name = parts[0].strip()
                    common_name = parts[1].strip()
                    japanese_name = parts[2].strip()
                elif len(parts) == 2:
                    scientific_name = parts[0].strip()
                    common_name = parts[1].strip()
                    japanese_name = parts[1].strip()

                display_name = japanese_name or common_name or scientific_name

                labels.append({
                    "index": i,
                    "raw_label": raw,
                    "scientific_name": scientific_name,
                    "common_name": common_name,
                    "japanese_name": japanese_name,
                    "display_name": display_name,
                    "keys": {
                        normalize_label_key(raw),
                        normalize_label_key(scientific_name),
                        normalize_label_key(common_name),
                        normalize_label_key(japanese_name),
                        normalize_label_key(display_name),
                    },
                })

        self.labels = labels

        print(
            "BirdNET labels loaded: "
            f"labels={len(self.labels)}, labels_file={labels_path.name}",
            flush=True,
        )

    def _label_allowed(self, label: dict) -> bool:
        if not self.allowed_scientific_names and not self.allowed_japanese_names:
            return True

        scientific_key = normalize_label_key(label.get("scientific_name"))
        japanese_key = normalize_label_key(label.get("japanese_name"))
        display_key = normalize_label_key(label.get("display_name"))

        if self.allowed_scientific_names and scientific_key in self.allowed_scientific_names:
            return True

        if self.allowed_japanese_names and (
            japanese_key in self.allowed_japanese_names
            or display_key in self.allowed_japanese_names
        ):
            return True

        return False

    def _find_package_meta_model(self) -> Path | None:
        try:
            import birdnet_analyzer

            root = Path(birdnet_analyzer.__file__).resolve().parent

            candidates = list(root.rglob("*MData*.tflite"))
            if candidates:
                return candidates[0]

            candidates = list(root.rglob("*Meta*.tflite"))
            if candidates:
                return candidates[0]

        except Exception:
            pass

        return None

    def _load_geo_scores(self) -> None:
        if not self.geo_enabled:
            self.geo_status = "disabled"
            self.geo_scores = {}
            print("BirdNET geo-season score: disabled", flush=True)
            return

        latitude = safe_float(self.birdnet_cfg.get("latitude"))
        longitude = safe_float(self.birdnet_cfg.get("longitude"))
        week = safe_float(self.birdnet_cfg.get("week"))

        if latitude is None or longitude is None or week is None:
            self.geo_status = "lat/lon/week missing"
            self.geo_scores = {}
            print(f"BirdNET geo-season score: {self.geo_status}", flush=True)
            return

        scores = self._load_geo_scores_with_birdnet_analyzer(latitude, longitude, week)

        if not scores:
            scores = self._load_geo_scores_with_tflite(latitude, longitude, week)

        if scores:
            self.geo_scores = scores
            self.geo_status = (
                f"enabled: lat={latitude}, lon={longitude}, week={int(week)}, "
                f"scores={len(scores)}"
            )
        else:
            self.geo_scores = {}
            self.geo_status = "enabled but unavailable"

        print(f"BirdNET geo-season score: {self.geo_status}", flush=True)

    def _load_geo_scores_with_birdnet_analyzer(
        self,
        latitude: float,
        longitude: float,
        week: float,
    ) -> dict[str, float]:
        """
        BirdNET-Analyzer の内部 API が利用できる場合に、地域・季節スコアを取得する。
        バージョン差を考慮し、複数の戻り値形式を受けられるようにしている。
        """
        try:
            from birdnet_analyzer import model_utils
        except Exception:
            return {}

        candidate_functions = []

        for name in [
            "run_geomodel",
            "predict_species_at_location_and_time",
            "predict_species_at_location",
        ]:
            func = getattr(model_utils, name, None)
            if callable(func):
                candidate_functions.append(func)

        for func in candidate_functions:
            try:
                result = self._call_geo_function(func, latitude, longitude, week)
                scores = self._parse_geo_result(result)
                if scores:
                    return scores
            except Exception:
                continue

        return {}

    def _call_geo_function(self, func, latitude: float, longitude: float, week: float):
        sig = inspect.signature(func)
        kwargs = {}

        for param_name in sig.parameters.keys():
            lower = param_name.lower()

            if lower in {"lat", "latitude"}:
                kwargs[param_name] = latitude
            elif lower in {"lon", "lng", "longitude"}:
                kwargs[param_name] = longitude
            elif lower in {"week", "week_of_year"}:
                kwargs[param_name] = int(week)
            elif lower in {"threshold", "species_threshold"}:
                kwargs[param_name] = 0.0
            elif lower in {"meta_model_path", "model_path", "model"}:
                meta_path = self._resolve_meta_model_path()
                if meta_path is not None:
                    kwargs[param_name] = str(meta_path)

        if kwargs:
            return func(**kwargs)

        try:
            return func(latitude, longitude, int(week), 0.0)
        except TypeError:
            return func(latitude, longitude, int(week))

    def _resolve_meta_model_path(self) -> Path | None:
        raw = self.birdnet_cfg.get("meta_model_path", "")
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path

        return self._find_package_meta_model()

    def _parse_geo_result(self, result) -> dict[str, float]:
        scores: dict[str, float] = {}

        if result is None:
            return scores

        if isinstance(result, dict):
            for key, value in result.items():
                score = clamp01(safe_float(value))
                if score is not None:
                    scores[normalize_label_key(str(key))] = score
            return scores

        try:
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                label_col = None
                score_col = None

                for c in result.columns:
                    if str(c).lower() in {"label", "species", "scientific_name", "name"}:
                        label_col = c
                    if str(c).lower() in {"confidence", "score", "probability", "likelihood"}:
                        score_col = c

                if label_col is not None and score_col is not None:
                    for _, row in result.iterrows():
                        score = clamp01(safe_float(row[score_col]))
                        if score is not None:
                            scores[normalize_label_key(str(row[label_col]))] = score
                    return scores
        except Exception:
            pass

        if isinstance(result, tuple) and len(result) >= 2:
            labels = result[0]
            values = result[1]
            try:
                for label, value in zip(labels, values):
                    score = clamp01(safe_float(value))
                    if score is not None:
                        scores[normalize_label_key(str(label))] = score
                return scores
            except Exception:
                pass

        if isinstance(result, list):
            for item in result:
                if isinstance(item, tuple) and len(item) >= 2:
                    score = clamp01(safe_float(item[1]))
                    if score is not None:
                        scores[normalize_label_key(str(item[0]))] = score
                elif isinstance(item, dict):
                    label = (
                        item.get("label")
                        or item.get("species")
                        or item.get("scientific_name")
                        or item.get("name")
                    )
                    value = (
                        item.get("confidence")
                        or item.get("score")
                        or item.get("probability")
                        or item.get("likelihood")
                    )
                    score = clamp01(safe_float(value))
                    if label and score is not None:
                        scores[normalize_label_key(str(label))] = score

            if scores:
                return scores

        arr = None
        try:
            arr = np.asarray(result, dtype=np.float32).reshape(-1)
        except Exception:
            arr = None

        if arr is not None and arr.size > 0:
            n = min(arr.size, len(self.labels))
            for i in range(n):
                score = clamp01(float(arr[i]))
                if score is None:
                    continue
                for key in self.labels[i]["keys"]:
                    if key:
                        scores[key] = score

        return scores

    def _load_geo_scores_with_tflite(
        self,
        latitude: float,
        longitude: float,
        week: float,
    ) -> dict[str, float]:
        """
        meta-model.tflite を直接実行するフォールバック。
        BirdNET-Analyzer のバージョン差で内部 API が呼べない場合でも、
        入力形式が単純なモデルであればスコアを取得できる。
        """
        meta_path = self._resolve_meta_model_path()
        if meta_path is None or not meta_path.exists():
            return {}

        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter
            except Exception:
                return {}

        try:
            interpreter = Interpreter(
                model_path=str(meta_path),
                num_threads=max(1, self.tflite_threads),
            )
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            if len(input_details) == 1:
                detail = input_details[0]
                shape = detail["shape"]
                dtype = detail["dtype"]

                values = np.array([[latitude, longitude, week]], dtype=np.float32)

                if len(shape) == 1:
                    values = values.reshape(shape)
                elif len(shape) == 2 and shape[-1] == 3:
                    values = values.reshape(shape)
                else:
                    values = np.resize(values, shape).astype(np.float32)

                interpreter.set_tensor(detail["index"], values.astype(dtype))

            elif len(input_details) >= 3:
                values = [latitude, longitude, week]
                for detail, value in zip(input_details[:3], values):
                    shape = detail["shape"]
                    dtype = detail["dtype"]
                    arr = np.array(value, dtype=np.float32).reshape(shape).astype(dtype)
                    interpreter.set_tensor(detail["index"], arr)

            else:
                return {}

            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]["index"])
            output = np.asarray(output, dtype=np.float32).reshape(-1)

            scores: dict[str, float] = {}
            n = min(output.size, len(self.labels))
            for i in range(n):
                score = clamp01(float(output[i]))
                if score is None:
                    continue

                for key in self.labels[i]["keys"]:
                    if key:
                        scores[key] = score

            return scores

        except Exception as e:
            print(f"BirdNET geo-season score direct tflite fallback failed: {e}", flush=True)
            return {}

    def _geo_score_for_label(self, label: dict) -> float | None:
        if not self.geo_enabled or not self.geo_scores:
            return None

        for key in label.get("keys", []):
            if key in self.geo_scores:
                return self.geo_scores[key]

        return None

    def _geo_label(self, score: float | None) -> str | None:
        if score is None:
            return None

        thresholds = self.birdnet_cfg.get("geo_score_thresholds", {}) or {}

        high = float(thresholds.get("high", 0.50))
        medium = float(thresholds.get("medium", 0.15))
        low = float(thresholds.get("low", 0.03))

        if score >= high:
            return "高"
        if score >= medium:
            return "中"
        if score >= low:
            return "低"
        return "要確認"

    def _normalize_model_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        BirdNET TFLite モデルの出力を、画面表示用の 0.0〜1.0 のスコアに正規化する。

        通常の BirdNET audio model は 0.0〜1.0 のスコアを返す想定だが、
        環境やモデル読み込み方法によって 1.0 を超える生スコアに見える値が
        得られることがあるため、表示用に安全側で丸める。

        ここでは以下の方針にする。
        - すべての値が 0.0〜1.0 に収まっている場合:
            そのまま clip する
        - それ以外の場合:
            logit 風の生スコアとみなして sigmoid 変換する
        """
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if scores.size == 0:
            return scores

        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            return np.zeros_like(scores, dtype=np.float32)

        min_score = float(np.min(finite))
        max_score = float(np.max(finite))

        if min_score >= 0.0 and max_score <= 1.0:
            return np.clip(scores, 0.0, 1.0).astype(np.float32)

        scores64 = scores.astype(np.float64)
        scores64 = np.clip(scores64, -50.0, 50.0)
        normalized = 1.0 / (1.0 + np.exp(-scores64))

        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def start(self) -> None:
        if self.running:
            update_state("birdnet", {
                "enabled": True,
                "running": True,
                "status": "running",
            })
            return

        self.running = True
        self.stop_event.clear()

        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(
                target=self._loop,
                name="BirdNETService",
                daemon=True,
            )
            self.thread.start()

        update_state("birdnet", {
            "enabled": True,
            "running": True,
            "status": "running",
            "capture_status": "ready",
            "analyze_status": "ready",
        })

    def stop(self) -> None:
        self.running = False
        update_state("birdnet", {
            "running": False,
            "status": "stopped",
        })

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.running:
                time.sleep(0.2)
                continue

            started = time.time()

            try:
                update_state("birdnet", {
                    "capture_status": "capturing",
                    "analyze_status": "waiting",
                })

                y = self.audio_engine.get_recent_audio(
                    seconds=self.window_seconds,
                    channel=self.birdnet_channel,
                )

                update_state("birdnet", {
                    "capture_status": "captured",
                    "analyze_status": "analyzing",
                })

                detection = self.analyze_audio(y)

                elapsed = time.time() - started
                detection["elapsed_seconds"] = elapsed

                update_state("birdnet", {
                    "running": True,
                    "status": "running",
                    "capture_status": "ready",
                    "analyze_status": "ready",
                    "last_elapsed_seconds": round(elapsed, 3),
                    "latest_detection": detection,
                })

                self._append_log(detection)

            except Exception as e:
                update_state("birdnet", {
                    "status": f"birdnet error: {e}",
                    "capture_status": "error",
                    "analyze_status": "error",
                })

            elapsed = time.time() - started
            sleep_seconds = max(0.05, self.interval_seconds - elapsed)
            time.sleep(sleep_seconds)

    def stop_thread(self) -> None:
        self.stop_event.set()
        self.running = False

    def _resample_if_needed(self, y: np.ndarray, target_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if y.size == target_samples:
            return y

        if y.size <= 1:
            return np.zeros(target_samples, dtype=np.float32)

        try:
            import librosa

            duration = y.size / float(self.sample_rate)
            target_sr = int(round(target_samples / max(duration, 1e-6)))
            y2 = librosa.resample(y, orig_sr=self.sample_rate, target_sr=target_sr)

            if y2.size > target_samples:
                y2 = y2[:target_samples]
            elif y2.size < target_samples:
                y2 = np.pad(y2, (0, target_samples - y2.size))

            return y2.astype(np.float32)

        except Exception:
            old_x = np.linspace(0.0, 1.0, y.size)
            new_x = np.linspace(0.0, 1.0, target_samples)
            y2 = np.interp(new_x, old_x, y).astype(np.float32)
            return y2

    def analyze_audio(self, y: np.ndarray) -> dict:
        if self.interpreter is None:
            raise RuntimeError("BirdNET model is not loaded.")

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        y = self._resample_if_needed(y, self.model_input_samples)

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0

        input_detail = self.input_details[0]
        input_shape = input_detail["shape"]
        input_dtype = input_detail["dtype"]

        x = y.astype(np.float32)

        if len(input_shape) == 2:
            x = x.reshape(1, -1)
        elif len(input_shape) == 3:
            x = x.reshape(1, -1, 1)
        else:
            x = np.reshape(x, input_shape)

        self.interpreter.set_tensor(input_detail["index"], x.astype(input_dtype))
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        scores = np.asarray(output, dtype=np.float32).reshape(-1)
        scores = self._normalize_model_scores(scores)

        top = self._make_top_candidates(scores)

        detection = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": None if top else "鳥の声を待っています",
            "top": top,
            "audio": {
                "peak": peak,
                "rms": rms,
                "channel": self.birdnet_channel,
                "window_seconds": self.window_seconds,
            },
            "geo": {
                "enabled": self.geo_enabled,
                "status": self.geo_status,
                "latitude": self.birdnet_cfg.get("latitude"),
                "longitude": self.birdnet_cfg.get("longitude"),
                "week": self.birdnet_cfg.get("week"),
            },
        }

        return detection

    def _make_top_candidates(self, scores: np.ndarray) -> list[dict]:
        if scores.size == 0:
            return []

        n = min(scores.size, len(self.labels))
        order = np.argsort(scores[:n])[::-1]

        top: list[dict] = []

        for idx in order:
            confidence = float(np.clip(scores[idx], 0.0, 1.0))

            if confidence < self.min_confidence:
                break

            label = self.labels[int(idx)]

            if not self._label_allowed(label):
                continue

            if confidence < self.display_confidence:
                continue

            geo_score = self._geo_score_for_label(label)
            geo_label = self._geo_label(geo_score)

            item = {
                "index": int(idx),
                "confidence": confidence,
                "scientific_name": label["scientific_name"],
                "common_name": label["common_name"],
                "japanese_name": label["japanese_name"],
                "display_name": label["display_name"],
                "raw_label": label["raw_label"],
                "geo_score": geo_score,
                "geo_label": geo_label,
                "geo_enabled": self.geo_enabled,
            }

            top.append(item)

            if len(top) >= 5:
                break

        return top

    def _append_log(self, detection: dict) -> None:
        log_csv = self.birdnet_cfg.get("log_csv")
        if not log_csv:
            return

        path = Path(log_csv)
        path.parent.mkdir(parents=True, exist_ok=True)

        top = detection.get("top") or []

        file_exists = path.exists()

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time",
                    "rank",
                    "display_name",
                    "japanese_name",
                    "scientific_name",
                    "common_name",
                    "confidence",
                    "geo_label",
                    "geo_score",
                    "geo_enabled",
                    "geo_status",
                    "audio_peak",
                    "audio_rms",
                    "elapsed_seconds",
                ],
            )

            if not file_exists:
                writer.writeheader()

            if not top:
                writer.writerow({
                    "time": detection.get("time"),
                    "rank": "",
                    "display_name": "",
                    "japanese_name": "",
                    "scientific_name": "",
                    "common_name": "",
                    "confidence": "",
                    "geo_label": "",
                    "geo_score": "",
                    "geo_enabled": self.geo_enabled,
                    "geo_status": self.geo_status,
                    "audio_peak": detection.get("audio", {}).get("peak"),
                    "audio_rms": detection.get("audio", {}).get("rms"),
                    "elapsed_seconds": detection.get("elapsed_seconds"),
                })
                return

            for rank, item in enumerate(top, start=1):
                geo_score = item.get("geo_score")
                writer.writerow({
                    "time": detection.get("time"),
                    "rank": rank,
                    "display_name": item.get("display_name"),
                    "japanese_name": item.get("japanese_name"),
                    "scientific_name": item.get("scientific_name"),
                    "common_name": item.get("common_name"),
                    "confidence": item.get("confidence"),
                    "geo_label": item.get("geo_label"),
                    "geo_score": "" if geo_score is None else geo_score,
                    "geo_enabled": self.geo_enabled,
                    "geo_status": self.geo_status,
                    "audio_peak": detection.get("audio", {}).get("peak"),
                    "audio_rms": detection.get("audio", {}).get("rms"),
                    "elapsed_seconds": detection.get("elapsed_seconds"),
                })