# state.py
from __future__ import annotations

import threading


STATE_LOCK = threading.Lock()

STATE = {
    "audio": {
        "status": "not started",
        "level_ch1": 0.0,
        "level_ch2": 0.0,
    },

    "birdnet": {
        "enabled": True,
        "running": False,
        "status": "ready",

        # BirdNET高速化版では、録音側と解析側が別スレッドで動くため、
        # 状態表示を分ける。
        "capture_status": "ready",
        "analyze_status": "ready",

        # 直近1回のBirdNET解析にかかった秒数。
        "last_elapsed_seconds": None,

        # 直近の判定結果。
        "latest_detection": None,
    },

    "mimic": {
        "enabled": False,
        "running": False,
        "status": "ready",
        "latest_score": None,
    },
}


def update_state(section: str, values: dict) -> None:
    with STATE_LOCK:
        if section not in STATE:
            STATE[section] = {}
        if not isinstance(STATE[section], dict):
            STATE[section] = {}
        STATE[section].update(values)


def get_state() -> dict:
    with STATE_LOCK:
        copied = {}
        for key, value in STATE.items():
            if isinstance(value, dict):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied