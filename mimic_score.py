# mimic_score.py
from __future__ import annotations

import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SR = 48000


def load_audio_raw(path: str | Path, sr: int = SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=sr, mono=True)

    if y.size == 0:
        return np.zeros(int(sr * 3), dtype=np.float32), sr

    return y.astype(np.float32), sr


def normalize_for_feature(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak

    return y.astype(np.float32)


def trim_or_pad(y: np.ndarray, seconds: float, sr: int = SR) -> np.ndarray:
    n = int(seconds * sr)

    if len(y) >= n:
        return y[:n].astype(np.float32)

    return np.pad(y, (0, n - len(y))).astype(np.float32)


def pad_to_length(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if len(y) >= n:
        return y[:n].astype(np.float32)

    return np.pad(y, (0, n - len(y))).astype(np.float32)


def audio_basic_stats(y: np.ndarray) -> dict:
    if y.size == 0:
        return {
            "peak": 0.0,
            "rms": 0.0,
        }

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))

    return {
        "peak": peak,
        "rms": rms,
    }


def voiced_ratio(
    y: np.ndarray,
    sr: int = SR,
    min_user_rms: float = 0.005,
) -> float:
    if y.size == 0:
        return 0.0

    env = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512,
    )[0]

    if env.size == 0:
        return 0.0

    threshold = max(min_user_rms * 1.5, 0.008)
    ratio = float(np.mean(env >= threshold))

    return ratio


def rms_envelope(y: np.ndarray, hop_length: int = 512) -> np.ndarray:
    env = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=hop_length,
    )[0]

    if env.size == 0:
        return np.zeros(1, dtype=np.float32)

    m = float(env.max())
    if m > 0:
        env = env / m

    return env.astype(np.float32)


def standardize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return np.zeros(1, dtype=np.float32)

    mean = float(np.mean(x))
    std = float(np.std(x))

    if std <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mean) / std).astype(np.float32)


def find_best_aligned_segment(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    seconds: float,
    sr: int = SR,
    hop_length: int = 512,
) -> tuple[np.ndarray, float, float]:
    """
    録音音声 usr_y の中から、お手本 ref_y に最も合う seconds 秒区間を探す。

    戻り値:
      aligned_user_y: 位置合わせ済み seconds 秒音声
      offset_seconds: usr_y の何秒目から切り出したか
      alignment_score: 0〜1程度の相関スコア
    """
    target_n = int(seconds * sr)

    ref_y = trim_or_pad(ref_y, seconds=seconds, sr=sr)
    usr_y = np.asarray(usr_y, dtype=np.float32).flatten()

    if len(usr_y) <= target_n:
        return pad_to_length(usr_y, target_n), 0.0, 0.0

    ref_env = rms_envelope(ref_y, hop_length=hop_length)
    usr_env = rms_envelope(usr_y, hop_length=hop_length)

    target_frames = max(1, int(round(target_n / hop_length)))

    if len(usr_env) <= target_frames:
        return pad_to_length(usr_y, target_n), 0.0, 0.0

    ref_env = pad_to_length(ref_env, target_frames)
    ref_env = standardize_1d(ref_env)

    best_i = 0
    best_score = -1.0

    max_i = len(usr_env) - target_frames

    for i in range(max_i + 1):
        seg_env = usr_env[i:i + target_frames]
        seg_env = standardize_1d(seg_env)

        denom = float(np.linalg.norm(ref_env) * np.linalg.norm(seg_env))
        if denom <= 1e-8:
            score = -1.0
        else:
            score = float(np.dot(ref_env, seg_env) / denom)

        if score > best_score:
            best_score = score
            best_i = i

    start_sample = int(best_i * hop_length)
    start_sample = max(0, min(start_sample, len(usr_y) - target_n))

    aligned = usr_y[start_sample:start_sample + target_n]
    aligned = pad_to_length(aligned, target_n)

    offset_seconds = start_sample / float(sr)
    alignment_score = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))

    return aligned.astype(np.float32), offset_seconds, alignment_score


def standardize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        return standardize_1d(x)

    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    std = np.where(std <= 1e-6, 1.0, std)

    return ((x - mean) / std).astype(np.float32)


def frame_normalize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        return standardize_feature(x)

    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std <= 1e-6, 1.0, std)

    return ((x - mean) / std).astype(np.float32)


def dtw_distance_normalized(ref: np.ndarray, usr: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float32)
    usr = np.asarray(usr, dtype=np.float32)

    if ref.size == 0 or usr.size == 0:
        return float("inf")

    if ref.ndim == 1:
        ref = ref.reshape(1, -1)

    if usr.ndim == 1:
        usr = usr.reshape(1, -1)

    feature_dim = max(ref.shape[0], usr.shape[0], 1)
    time_dim = max(ref.shape[1], usr.shape[1], 1)

    try:
        D, _ = librosa.sequence.dtw(
            X=ref,
            Y=usr,
            metric="euclidean",
        )
        raw_dist = float(D[-1, -1])
    except Exception:
        return float("inf")

    normalized = raw_dist / time_dim / math.sqrt(feature_dim)
    return float(normalized)


def dtw_similarity_score(
    ref: np.ndarray,
    usr: np.ndarray,
    strictness: float = 1.5,
    default_score: float = 0.0,
) -> float:
    dist = dtw_distance_normalized(ref, usr)

    if not np.isfinite(dist):
        return default_score

    score = 100.0 * math.exp(-strictness * dist)
    return float(np.clip(score, 0, 100))


def cosine_similarity_score(
    ref: np.ndarray,
    usr: np.ndarray,
    min_score: float = 0.0,
) -> float:
    ref = np.asarray(ref, dtype=np.float32).flatten()
    usr = np.asarray(usr, dtype=np.float32).flatten()

    n = min(len(ref), len(usr))
    if n <= 0:
        return min_score

    ref = ref[:n]
    usr = usr[:n]

    ref = ref - float(np.mean(ref))
    usr = usr - float(np.mean(usr))

    ref_norm = float(np.linalg.norm(ref))
    usr_norm = float(np.linalg.norm(usr))

    if ref_norm <= 1e-8 or usr_norm <= 1e-8:
        return min_score

    sim = float(np.dot(ref, usr) / (ref_norm * usr_norm))
    score = 100.0 * (sim + 1.0) / 2.0

    return float(np.clip(score, min_score, 100))


def rhythm_score(ref_y: np.ndarray, usr_y: np.ndarray) -> float:
    ref = rms_envelope(ref_y)
    usr = rms_envelope(usr_y)

    return dtw_similarity_score(
        ref.reshape(1, -1),
        usr.reshape(1, -1),
        strictness=4.0,
        default_score=0.0,
    )


def mel_spectrogram_db(
    y: np.ndarray,
    sr: int = SR,
    n_mels: int = 64,
    fmin: int = 150,
    fmax: int = 15000,
    db_min: float = -60.0,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    db = librosa.power_to_db(mel, ref=np.max)
    db = np.maximum(db, db_min)

    return db.astype(np.float32)


def mel_dtw_score(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    sr: int = SR,
    strictness: float = 1.35,
) -> float:
    ref_db = mel_spectrogram_db(ref_y, sr=sr)
    usr_db = mel_spectrogram_db(usr_y, sr=sr)

    ref_feat = frame_normalize_feature(ref_db)
    usr_feat = frame_normalize_feature(usr_db)

    return dtw_similarity_score(
        ref_feat,
        usr_feat,
        strictness=strictness,
        default_score=0.0,
    )


def mel_cosine_score(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    sr: int = SR,
) -> float:
    ref_db = mel_spectrogram_db(ref_y, sr=sr)
    usr_db = mel_spectrogram_db(usr_y, sr=sr)

    return cosine_similarity_score(ref_db, usr_db, min_score=0.0)


def spectral_profile_score(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    sr: int = SR,
) -> float:
    ref_db = mel_spectrogram_db(ref_y, sr=sr)
    usr_db = mel_spectrogram_db(usr_y, sr=sr)

    ref_profile = np.mean(ref_db, axis=1)
    usr_profile = np.mean(usr_db, axis=1)

    ref_profile = standardize_feature(ref_profile)
    usr_profile = standardize_feature(usr_profile)

    return cosine_similarity_score(
        ref_profile,
        usr_profile,
        min_score=0.0,
    )


def mfcc_dtw_score(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    sr: int = SR,
    strictness: float = 1.0,
) -> float:
    ref_mfcc = librosa.feature.mfcc(
        y=ref_y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        fmin=150,
        fmax=15000,
    )

    usr_mfcc = librosa.feature.mfcc(
        y=usr_y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        fmin=150,
        fmax=15000,
    )

    ref_mfcc = ref_mfcc[1:, :]
    usr_mfcc = usr_mfcc[1:, :]

    ref_feat = standardize_feature(ref_mfcc)
    usr_feat = standardize_feature(usr_mfcc)

    return dtw_similarity_score(
        ref_feat,
        usr_feat,
        strictness=strictness,
        default_score=0.0,
    )


def spectral_centroid_score(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    sr: int = SR,
    strictness: float = 2.0,
) -> float:
    ref_centroid = librosa.feature.spectral_centroid(
        y=ref_y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
    )[0]

    usr_centroid = librosa.feature.spectral_centroid(
        y=usr_y,
        sr=sr,
        n_fft=2048,
        hop_length=512,
    )[0]

    if ref_centroid.size == 0 or usr_centroid.size == 0:
        return 0.0

    ref_centroid = ref_centroid / max(sr / 2.0, 1.0)
    usr_centroid = usr_centroid / max(sr / 2.0, 1.0)

    ref_feat = standardize_feature(ref_centroid).reshape(1, -1)
    usr_feat = standardize_feature(usr_centroid).reshape(1, -1)

    return dtw_similarity_score(
        ref_feat,
        usr_feat,
        strictness=strictness,
        default_score=0.0,
    )


def shape_score(
    ref_y: np.ndarray,
    usr_y: np.ndarray,
    sr: int = SR,
    cfg: dict | None = None,
) -> float:
    cfg = cfg or {}

    shape_strictness = float(cfg.get("shape_strictness", 1.35))

    mel_dtw_weight = float(cfg.get("mel_dtw_weight", 0.22))
    mfcc_weight = float(cfg.get("mfcc_weight", 0.18))
    centroid_weight = float(cfg.get("centroid_weight", 0.10))
    mel_cosine_weight = float(cfg.get("mel_cosine_weight", 0.27))
    spectral_profile_weight = float(cfg.get("spectral_profile_weight", 0.23))

    mel_dtw = mel_dtw_score(
        ref_y,
        usr_y,
        sr=sr,
        strictness=shape_strictness,
    )

    mfcc = mfcc_dtw_score(
        ref_y,
        usr_y,
        sr=sr,
        strictness=max(0.8, shape_strictness * 0.65),
    )

    centroid = spectral_centroid_score(
        ref_y,
        usr_y,
        sr=sr,
        strictness=max(1.5, shape_strictness * 1.2),
    )

    mel_cos = mel_cosine_score(
        ref_y,
        usr_y,
        sr=sr,
    )

    profile = spectral_profile_score(
        ref_y,
        usr_y,
        sr=sr,
    )

    total_weight = (
        mel_dtw_weight +
        mfcc_weight +
        centroid_weight +
        mel_cosine_weight +
        spectral_profile_weight
    )

    if total_weight <= 0:
        return 0.0

    score = (
        mel_dtw_weight * mel_dtw +
        mfcc_weight * mfcc +
        centroid_weight * centroid +
        mel_cosine_weight * mel_cos +
        spectral_profile_weight * profile
    ) / total_weight

    return float(np.clip(score, 0, 100))


def pitch_motion_score(ref_y: np.ndarray, usr_y: np.ndarray, sr: int = SR) -> float:
    def curve(y: np.ndarray) -> np.ndarray:
        try:
            f0, _, _ = librosa.pyin(
                y,
                fmin=100,
                fmax=3000,
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
        except Exception:
            return np.zeros(1, dtype=np.float32)

        f0 = np.nan_to_num(f0, nan=0.0)

        if f0.size == 0:
            return np.zeros(1, dtype=np.float32)

        voiced = f0 > 0
        if float(np.mean(voiced)) < 0.05:
            return np.zeros_like(f0, dtype=np.float32)

        nonzero = f0[voiced]
        median = float(np.median(nonzero)) if nonzero.size else 0.0

        if median > 0:
            f0 = f0 / median

        f0 = np.clip(f0, 0.0, 3.0)

        return f0.astype(np.float32)

    ref = curve(ref_y)
    usr = curve(usr_y)

    n = min(len(ref), len(usr))
    if n == 0:
        return 0.0

    ref = ref[:n]
    usr = usr[:n]

    if np.all(ref == 0) or np.all(usr == 0):
        return 20.0

    corr = np.corrcoef(ref, usr)[0, 1]
    if np.isnan(corr):
        return 20.0

    score = 100.0 * (corr + 1.0) / 2.0
    return float(np.clip(score, 0, 100))


def loudness_score(ref_y: np.ndarray, usr_y: np.ndarray) -> float:
    ref = float(np.sqrt(np.mean(ref_y ** 2)))
    usr = float(np.sqrt(np.mean(usr_y ** 2)))

    if ref <= 0 or usr <= 0:
        return 0.0

    ratio = min(ref, usr) / max(ref, usr)
    return float(np.clip(100.0 * ratio, 0, 100))


def energy_score(
    user_peak: float,
    user_rms: float,
    cfg: dict | None = None,
) -> float:
    cfg = cfg or {}

    target_rms = float(cfg.get("energy_target_rms", 0.030))
    target_peak = float(cfg.get("energy_target_peak", 0.180))

    if target_rms <= 0:
        target_rms = 0.030

    if target_peak <= 0:
        target_peak = 0.180

    rms_score = 100.0 * min(user_rms / target_rms, 1.0)
    peak_score = 100.0 * min(user_peak / target_peak, 1.0)

    score = 0.70 * rms_score + 0.30 * peak_score

    return float(np.clip(score, 0, 100))


def choose_title(score: float, title_bands: list[dict] | None = None) -> str:
    if not title_bands:
        title_bands = [
            {"min_score": 95, "title": "本物と会話できる鳥人級！"},
            {"min_score": 90, "title": "森の通訳者級！"},
            {"min_score": 80, "title": "鳥の友達級！"},
            {"min_score": 70, "title": "かなり鳥っぽい級！"},
            {"min_score": 60, "title": "鳥まねチャレンジャー級！"},
            {"min_score": 50, "title": "もう少しで鳥語級！"},
            {"min_score": 0, "title": "まだ人間だった級！"},
        ]

    bands = sorted(
        title_bands,
        key=lambda x: float(x.get("min_score", 0)),
        reverse=True,
    )

    for band in bands:
        if score >= float(band.get("min_score", 0)):
            return str(band.get("title", "鳥まねチャレンジャー級！"))

    return "鳥まねチャレンジャー級！"


def make_strict_comment(
    total: float,
    rhythm: float,
    shape: float,
    pitch: float,
    cfg: dict | None = None,
) -> str:
    cfg = cfg or {}

    shape_comment_min_shape = float(cfg.get("shape_comment_min_shape", 70.0))
    shape_comment_min_total = float(cfg.get("shape_comment_min_total", 68.0))

    if total >= 85:
        return "すごい！鳥にかなり近い鳴き方です。"

    if rhythm >= 82 and total >= 70:
        return "リズムがとてもよく似ています。"

    if pitch >= 80 and total >= 70:
        return "声の上がり下がりがよく似ています。"

    if shape >= shape_comment_min_shape and total >= shape_comment_min_total:
        return "音の形がよく似ています。"

    if total >= 60:
        return "いい感じです。もう一度やるとさらに近づくかも。"

    return "鳥の声をよく聞いて、リズムをまねしてみよう。"


def make_game_comment(
    total: float,
    rhythm: float,
    pitch: float,
    shape: float,
    energy: float,
) -> str:
    if total >= 90:
        return "すごい！鳥とおしゃべりできそうな鳴きまねです。"

    if total >= 80:
        return "かなり鳥っぽい！特徴をよくつかめています。"

    if rhythm >= 80 and total >= 65:
        return "タイミングがばっちり！次は声の高さもまねしてみよう。"

    if pitch >= 75 and total >= 65:
        return "声の高さがいい感じ！リズムを合わせるともっと鳥っぽくなります。"

    if shape >= 70 and total >= 65:
        return "鳴き声の形が似ています！もう少し元気に鳴くと高得点かも。"

    if energy < 45:
        return "もう少し大きな声で鳴くと点が上がりそうです。"

    if total >= 60:
        return "いい感じ！鳥の鳴き声の特徴をもう一度聞いてみよう。"

    return "鳥の声をよく聞いて、リズムと高さをまねしてみよう。"


def strict_total_score(
    rhythm: float,
    shape: float,
    pitch: float,
    loud: float,
    cfg: dict | None = None,
) -> float:
    cfg = cfg or {}

    rhythm_weight = float(cfg.get("rhythm_weight", 0.36))
    shape_weight = float(cfg.get("shape_weight", 0.43))
    pitch_weight = float(cfg.get("pitch_weight", 0.16))
    loudness_weight = float(cfg.get("loudness_weight", 0.05))

    weight_sum = (
        rhythm_weight +
        shape_weight +
        pitch_weight +
        loudness_weight
    )

    if weight_sum <= 0:
        return 0.0

    total = (
        rhythm_weight * rhythm +
        shape_weight * shape +
        pitch_weight * pitch +
        loudness_weight * loud
    ) / weight_sum

    return float(np.clip(total, 0, 100))


def game_total_score(
    rhythm: float,
    shape: float,
    pitch: float,
    energy: float,
    cfg: dict | None = None,
) -> float:
    cfg = cfg or {}

    rhythm_weight = float(cfg.get("rhythm_weight", 0.40))
    pitch_weight = float(cfg.get("pitch_weight", 0.30))
    shape_weight = float(cfg.get("shape_weight", 0.15))
    energy_weight = float(cfg.get("energy_weight", 0.15))
    min_valid_score = float(cfg.get("min_valid_score", 35.0))

    weight_sum = (
        rhythm_weight +
        pitch_weight +
        shape_weight +
        energy_weight
    )

    if weight_sum <= 0:
        return 0.0

    total = (
        rhythm_weight * rhythm +
        pitch_weight * pitch +
        shape_weight * shape +
        energy_weight * energy
    ) / weight_sum

    total = max(total, min_valid_score)

    return float(np.clip(total, 0, 100))


def invalid_result(
    reason: str,
    user_peak: float,
    user_rms: float,
    user_voiced_ratio: float,
    alignment_offset_seconds: float = 0.0,
    alignment_score: float = 0.0,
) -> dict:
    return {
        "valid": False,
        "reason": reason,
        "total": 0.0,
        "game_total": 0.0,
        "strict_total": 0.0,
        "title": "もう一度チャレンジ！",
        "comment": reason,
        "rhythm": 0.0,
        "shape": 0.0,
        "pitch": 0.0,
        "loudness": 0.0,
        "energy": 0.0,
        "alignment_offset_seconds": round(float(alignment_offset_seconds), 3),
        "alignment_score": round(float(alignment_score), 3),
        "user_peak": round(float(user_peak), 6),
        "user_rms": round(float(user_rms), 6),
        "user_voiced_ratio": round(float(user_voiced_ratio), 4),
    }


def mimic_score(
    reference_wav: str | Path,
    user_wav: str | Path,
    seconds: float = 3.0,
    min_user_rms: float = 0.005,
    min_user_peak: float = 0.030,
    min_voiced_ratio: float = 0.08,
    scoring_mode: str = "both",
    strict_scoring_cfg: dict | None = None,
    game_scoring_cfg: dict | None = None,
    title_bands: list[dict] | None = None,
    aligned_user_wav: str | Path | None = None,
) -> dict:
    strict_scoring_cfg = strict_scoring_cfg or {}
    game_scoring_cfg = game_scoring_cfg or {}

    scoring_mode = str(scoring_mode or "both").lower()
    if scoring_mode not in {"strict", "game", "both"}:
        scoring_mode = "both"

    ref_raw, sr = load_audio_raw(reference_wav)
    usr_raw, _ = load_audio_raw(user_wav, sr=sr)

    ref_raw = trim_or_pad(ref_raw, seconds=seconds, sr=sr)

    aligned_usr_raw, alignment_offset_seconds, alignment_score = find_best_aligned_segment(
        ref_y=ref_raw,
        usr_y=usr_raw,
        seconds=seconds,
        sr=sr,
    )

    if aligned_user_wav is not None:
        aligned_user_wav = Path(aligned_user_wav)
        aligned_user_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(aligned_user_wav, aligned_usr_raw, sr, subtype="PCM_16")

    user_stats = audio_basic_stats(aligned_usr_raw)
    user_peak = user_stats["peak"]
    user_rms = user_stats["rms"]

    user_voiced_ratio = voiced_ratio(
        aligned_usr_raw,
        sr=sr,
        min_user_rms=min_user_rms,
    )

    if user_peak < min_user_peak:
        return invalid_result(
            reason="声が小さすぎるようです。マイクに向かってもう少し大きくまねしてみよう。",
            user_peak=user_peak,
            user_rms=user_rms,
            user_voiced_ratio=user_voiced_ratio,
            alignment_offset_seconds=alignment_offset_seconds,
            alignment_score=alignment_score,
        )

    if user_rms < min_user_rms:
        return invalid_result(
            reason="鳴き真似の声がほとんど入っていないようです。もう一度チャレンジしてみよう。",
            user_peak=user_peak,
            user_rms=user_rms,
            user_voiced_ratio=user_voiced_ratio,
            alignment_offset_seconds=alignment_offset_seconds,
            alignment_score=alignment_score,
        )

    if user_voiced_ratio < min_voiced_ratio:
        return invalid_result(
            reason="鳴き真似の時間が短いようです。3秒間、鳥の声をまねしてみよう。",
            user_peak=user_peak,
            user_rms=user_rms,
            user_voiced_ratio=user_voiced_ratio,
            alignment_offset_seconds=alignment_offset_seconds,
            alignment_score=alignment_score,
        )

    ref_y = normalize_for_feature(ref_raw)
    usr_y = normalize_for_feature(aligned_usr_raw)

    rhythm = rhythm_score(ref_y, usr_y)
    shape = shape_score(
        ref_y,
        usr_y,
        sr=sr,
        cfg=strict_scoring_cfg,
    )
    pitch = pitch_motion_score(ref_y, usr_y, sr=sr)
    loud = loudness_score(ref_raw, aligned_usr_raw)
    energy = energy_score(
        user_peak=user_peak,
        user_rms=user_rms,
        cfg=game_scoring_cfg,
    )

    strict_total = strict_total_score(
        rhythm=rhythm,
        shape=shape,
        pitch=pitch,
        loud=loud,
        cfg=strict_scoring_cfg,
    )

    game_total = game_total_score(
        rhythm=rhythm,
        shape=shape,
        pitch=pitch,
        energy=energy,
        cfg=game_scoring_cfg,
    )

    if scoring_mode == "strict":
        display_total = strict_total
        comment = make_strict_comment(
            total=strict_total,
            rhythm=rhythm,
            shape=shape,
            pitch=pitch,
            cfg=strict_scoring_cfg,
        )
    else:
        display_total = game_total
        comment = make_game_comment(
            total=game_total,
            rhythm=rhythm,
            pitch=pitch,
            shape=shape,
            energy=energy,
        )

    title = choose_title(display_total, title_bands=title_bands)

    return {
        "valid": True,
        "reason": "",
        "total": round(float(display_total), 1),
        "game_total": round(float(game_total), 1),
        "strict_total": round(float(strict_total), 1),
        "title": title,
        "comment": comment,
        "rhythm": round(float(rhythm), 1),
        "shape": round(float(shape), 1),
        "pitch": round(float(pitch), 1),
        "loudness": round(float(loud), 1),
        "energy": round(float(energy), 1),
        "alignment_offset_seconds": round(float(alignment_offset_seconds), 3),
        "alignment_score": round(float(alignment_score), 3),
        "user_peak": round(float(user_peak), 6),
        "user_rms": round(float(user_rms), 6),
        "user_voiced_ratio": round(float(user_voiced_ratio), 4),
    }