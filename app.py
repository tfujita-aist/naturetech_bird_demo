# app.py
from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from audio_engine import AudioEngine
from birdnet_service import BirdNETService
from mimic_service import MimicService
from state import get_state, update_state


Path("reference_sounds").mkdir(parents=True, exist_ok=True)
Path("work/mimic_recordings").mkdir(parents=True, exist_ok=True)

APP = FastAPI()

APP.mount(
    "/reference_sounds",
    StaticFiles(directory="reference_sounds"),
    name="reference_sounds",
)

APP.mount(
    "/mimic_recordings",
    StaticFiles(directory="work/mimic_recordings"),
    name="mimic_recordings",
)

TEMPLATES = Jinja2Templates(directory="templates")

CONFIG = None
AUDIO_ENGINE = None
BIRDNET_SERVICE = None
MIMIC_SERVICE = None


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    paths = [
        cfg["birdnet"]["input_dir"],
        cfg["birdnet"]["output_dir"],
        cfg["mimic"]["output_dir"],
        Path(cfg["birdnet"]["log_csv"]).parent,
        Path(cfg["mimic"]["log_csv"]).parent,
        "templates",
        "reference_sounds",
    ]

    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


@APP.on_event("startup")
async def on_startup():
    global CONFIG, AUDIO_ENGINE, BIRDNET_SERVICE, MIMIC_SERVICE

    CONFIG = load_config("config.yaml")
    ensure_dirs(CONFIG)

    AUDIO_ENGINE = AudioEngine(CONFIG)
    AUDIO_ENGINE.start()

    BIRDNET_SERVICE = BirdNETService(CONFIG, AUDIO_ENGINE)
    update_state("birdnet", {
        "enabled": True,
        "running": False,
        "status": "ready",
    })

    MIMIC_SERVICE = MimicService(CONFIG, AUDIO_ENGINE)

    # 鳴き真似チャレンジはアプリ起動時から常に利用可能にする。
    MIMIC_SERVICE.start()
    update_state("mimic", {
        "enabled": True,
        "running": False,
        "status": "ready",
    })


@APP.on_event("shutdown")
async def on_shutdown():
    if BIRDNET_SERVICE is not None:
        BIRDNET_SERVICE.stop()

    if MIMIC_SERVICE is not None:
        MIMIC_SERVICE.stop()

    if AUDIO_ENGINE is not None:
        AUDIO_ENGINE.stop()


@APP.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {"config": CONFIG},
    )


@APP.get("/index.html", response_class=HTMLResponse)
async def index_html(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {"config": CONFIG},
    )


@APP.get("/birdnet", response_class=HTMLResponse)
async def birdnet_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "birdnet.html",
        {"config": CONFIG},
    )


@APP.get("/mimic", response_class=HTMLResponse)
async def mimic_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "mimic.html",
        {"config": CONFIG},
    )


@APP.get("/api/state")
async def api_state():
    return JSONResponse(get_state())


@APP.post("/api/birdnet/start")
async def api_birdnet_start():
    if BIRDNET_SERVICE is None:
        return JSONResponse(
            {"error": "BirdNET service is not initialized"},
            status_code=500,
        )

    BIRDNET_SERVICE.start()
    return JSONResponse({"ok": True, "message": "birdnet started"})


@APP.post("/api/birdnet/stop")
async def api_birdnet_stop():
    if BIRDNET_SERVICE is None:
        return JSONResponse(
            {"error": "BirdNET service is not initialized"},
            status_code=500,
        )

    BIRDNET_SERVICE.stop()
    return JSONResponse({"ok": True, "message": "birdnet stopping"})


@APP.post("/api/mimic/start")
async def api_mimic_start():
    if MIMIC_SERVICE is None:
        return JSONResponse(
            {"error": "Mimic service is not initialized"},
            status_code=500,
        )

    MIMIC_SERVICE.start()
    return JSONResponse({"ok": True, "message": "mimic started"})


@APP.post("/api/mimic/stop")
async def api_mimic_stop():
    if MIMIC_SERVICE is None:
        return JSONResponse(
            {"error": "Mimic service is not initialized"},
            status_code=500,
        )

    MIMIC_SERVICE.stop()
    return JSONResponse({"ok": True, "message": "mimic stopped"})


@APP.post("/api/mimic/{species_key}")
async def api_mimic(
    species_key: str,
    reference_index: int | None = Query(default=None),
):
    if MIMIC_SERVICE is None:
        return JSONResponse(
            {"error": "Mimic service is not initialized"},
            status_code=500,
        )

    try:
        result = await asyncio.to_thread(
            MIMIC_SERVICE.record_and_score,
            species_key,
            reference_index,
        )
        return JSONResponse(result)

    except Exception as e:
        update_state("mimic", {
            "running": False,
            "status": f"mimic error: {e}",
        })
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        APP,
        host="127.0.0.1",
        port=8000,
    )