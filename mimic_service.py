# mimic_service.py
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf

from mimic_score import mimic_score
from state import get_state, update_state


class MimicService:
    def __init__(self, cfg: dict, audio_engine):
        self.cfg = cfg
        self.audio_engine = audio_engine

        update_state("mimic", {
            "enabled": False,
            "running": False,
            "status": "ready",
            "latest_score": None,
        })

    def start(self) -> None:
        update_state("mimic", {
            "enabled": True,
            "running": False,
            "status": "ready",
        })

    def stop(self) -> None:
        update_state("mimic", {
            "enabled": False,
            "running": False,
            "status": "stopped",
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

    def get_reference_wavs(self, species: dict) -> list[str]:
        wavs = species.get("reference_wavs")

        if isinstance(wavs, list) and wavs:
            return [str(x) for x in wavs]

        reference_audio = species.get("reference_audio")
        if reference_audio:
            return [str(reference_audio)]

        reference_wav = species.get("reference_wav")
        if reference_wav:
            return [str(reference_wav)]

        return []

    def resolve_reference_audio(
        self,
        species_key: str,
        species: dict,
        reference_index: int | None,
    ) -> tuple[str, int]:
        reference_wavs = self.get_reference_wavs(species)

        if not reference_wavs:
            raise RuntimeError(f"No reference audio configured for {species_key}")

        if reference_index is None:
            reference_index = 0

        reference_index = int(reference_index)

        if reference_index < 0 or reference_index >= len(reference_wavs):
            raise ValueError(
                f"Invalid reference_index={reference_index} for {species_key}. "
                f"Available range: 0..{len(reference_wavs) - 1}"
            )

        return reference_wavs[reference_index], reference_index

    def file_url_from_path(
        self,
        file_path: str | Path | None,
        base_dir: str | Path,
        url_prefix: str,
    ) -> str:
        if not file_path:
            return ""

        file_path = Path(file_path)
        base_dir = Path(base_dir)

        try:
            rel = file_path.relative_to(base_dir)
            return url_prefix.rstrip("/") + "/" + str(rel).replace("\\", "/")
        except ValueError:
            return ""

    def record_and_score(
        self,
        species_key: str,
        reference_index: int | None = None,
    ) -> dict:
        state = get_state()

        if not state["mimic"]["enabled"]:
            raise RuntimeError("鳴き真似機能が停止中です。")

        cfg = self.cfg

        if species_key not in cfg["mimic"]["species"]:
            raise ValueError(f"Unknown species key: {species_key}")

        species = cfg["mimic"]["species"][species_key]

        reference_audio, resolved_reference_index = self.resolve_reference_audio(
            species_key=species_key,
            species=species,
            reference_index=reference_index,
        )

        score_seconds = float(cfg["mimic"]["record_seconds"])
        alignment_extra_seconds = float(cfg["mimic"].get("alignment_extra_seconds", 0.0))
        capture_seconds = score_seconds + max(0.0, alignment_extra_seconds)

        ch = int(cfg["audio"]["mimic_channel"])
        sr = int(cfg["audio"]["sample_rate"])

        update_state("mimic", {
            "running": True,
            "status": (
                f"recording {species_key} "
                f"reference_index={resolved_reference_index}"
            ),
        })

        time.sleep(capture_seconds)

        y = self.audio_engine.get_recent_audio(capture_seconds, ch)

        out_dir = Path(cfg["mimic"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = out_dir / f"mimic_{species_key}_{ts}.wav"
        aligned_wav_path = out_dir / f"mimic_{species_key}_{ts}_aligned.wav"

        sf.write(wav_path, y, sr, subtype="PCM_16")

        scoring_mode = str(cfg["mimic"].get("scoring_mode", "both")).lower()

        result = mimic_score(
            reference_wav=reference_audio,
            user_wav=wav_path,
            seconds=score_seconds,
            min_user_rms=float(cfg["mimic"].get("min_user_rms", 0.005)),
            min_user_peak=float(cfg["mimic"].get("min_user_peak", 0.030)),
            min_voiced_ratio=float(cfg["mimic"].get("min_voiced_ratio", 0.08)),
            scoring_mode=scoring_mode,
            strict_scoring_cfg=cfg["mimic"].get("strict_scoring", {}),
            game_scoring_cfg=cfg["mimic"].get("game_scoring", {}),
            title_bands=species.get("title_bands", cfg["mimic"].get("title_bands", [])),
            aligned_user_wav=aligned_wav_path,
        )

        reference_audio_url = "/" + reference_audio.replace("\\", "/").lstrip("/")

        wav_url = self.file_url_from_path(
            file_path=wav_path,
            base_dir=out_dir,
            url_prefix="/mimic_recordings",
        )

        aligned_wav_url = self.file_url_from_path(
            file_path=aligned_wav_path,
            base_dir=out_dir,
            url_prefix="/mimic_recordings",
        )

        row = {
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "species_key": species_key,
            "japanese_name": species["japanese_name"],
            "reference_index": resolved_reference_index,
            "reference_audio": reference_audio,
            "reference_audio_url": reference_audio_url,
            "scoring_mode": scoring_mode,

            "display_total": result["total"],
            "game_total": result["game_total"],
            "strict_total": result["strict_total"],

            "title": result["title"],
            "comment": result["comment"],

            "rhythm": result["rhythm"],
            "shape": result["shape"],
            "pitch": result["pitch"],
            "loudness": result["loudness"],
            "energy": result["energy"],

            "valid": result["valid"],
            "reason": result["reason"],

            "alignment_offset_seconds": result.get("alignment_offset_seconds", 0.0),
            "alignment_score": result.get("alignment_score", 0.0),

            "user_peak": result["user_peak"],
            "user_rms": result["user_rms"],
            "user_voiced_ratio": result["user_voiced_ratio"],

            "wav": str(wav_path),
            "wav_url": wav_url,
            "aligned_wav": str(aligned_wav_path),
            "aligned_wav_url": aligned_wav_url,
        }

        self.append_csv(cfg["mimic"]["log_csv"], row)

        response = row.copy()

        update_state("mimic", {
            "running": False,
            "status": "scored" if result["valid"] else "invalid",
            "latest_score": response,
        })

        return response