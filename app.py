# app.py
from __future__ import annotations

import asyncio
import socket
import subprocess
import time
from io import BytesIO
from pathlib import Path

import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

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


def is_usable_lan_ip(ip: str) -> bool:
    if not ip:
        return False

    ip = ip.strip()

    if ip.startswith("127."):
        return False

    if ip.startswith("169.254."):
        return False

    return True


def get_ip_from_ifaddr(interface_name: str) -> str | None:
    try:
        result = subprocess.run(
            ["ipconfig", "getifaddr", interface_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.0,
            check=False,
        )

        ip = result.stdout.strip()

        if result.returncode == 0 and is_usable_lan_ip(ip):
            return ip

    except Exception:
        pass

    return None


def get_lan_ip() -> str:
    for interface_name in ["en0", "en1"]:
        ip = get_ip_from_ifaddr(interface_name)
        if ip:
            return ip

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if is_usable_lan_ip(ip):
                return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if is_usable_lan_ip(ip):
            return ip
    except Exception:
        pass

    return "127.0.0.1"


def get_control_url() -> str:
    ip = get_lan_ip()
    return f"http://{ip}:8000/control"


def print_access_urls() -> None:
    lan_ip = get_lan_ip()

    print("", flush=True)
    print("============================================================", flush=True)
    print("NATURE TECH! 2026 鳥の声 AI デモ", flush=True)
    print("============================================================", flush=True)
    print("", flush=True)
    print("モニター側のブラウザでは、以下の URL を開いてください。", flush=True)
    print("", flush=True)
    print("  入口画面:", flush=True)
    print("    http://localhost:8000/index.html", flush=True)
    print("", flush=True)
    print("  鳥の声AI判定:", flush=True)
    print("    http://localhost:8000/birdnet", flush=True)
    print("", flush=True)
    print("  鳥の鳴きまねチャレンジ:", flush=True)
    print("    http://localhost:8000/mimic", flush=True)
    print("", flush=True)
    print("操作パネルは、外部デバイス側で以下を開いてください。", flush=True)
    print("", flush=True)
    print(f"    http://{lan_ip}:8000/control", flush=True)
    print("", flush=True)
    print("============================================================", flush=True)
    print("", flush=True)


def template_context(request: Request) -> dict:
    return {
        "request": request,
        "config": CONFIG,
        "control_url": get_control_url(),
        "control_qr_url": "/control_qr.svg",
    }


def make_mimic_command(species_key: str, source: str) -> dict:
    command_id = int(time.time() * 1000)

    return {
        "id": command_id,
        "type": "mimic",
        "species_key": species_key,
        "source": source,
        "created_at_ms": command_id,
    }


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
        "capture_status": "ready",
        "analyze_status": "ready",
        "last_elapsed_seconds": None,
        "latest_detection": None,
    })

    MIMIC_SERVICE = MimicService(CONFIG, AUDIO_ENGINE)

    MIMIC_SERVICE.start()
    update_state("mimic", {
        "enabled": True,
        "running": False,
        "status": "ready",
        "active_species_key": None,
        "latest_score": None,
    })

    update_state("control", {
        "mimic_command": None,
        "active_species_key": None,
    })

    print_access_urls()


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
        template_context(request),
    )


@APP.get("/index.html", response_class=HTMLResponse)
async def index_html(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        template_context(request),
    )


@APP.get("/birdnet", response_class=HTMLResponse)
async def birdnet_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "birdnet.html",
        template_context(request),
    )


@APP.get("/mimic", response_class=HTMLResponse)
async def mimic_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "mimic.html",
        template_context(request),
    )


@APP.get("/control", response_class=HTMLResponse)
async def control_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "control.html",
        template_context(request),
    )


@APP.get("/api/control-url")
async def api_control_url():
    return JSONResponse({
        "control_url": get_control_url(),
        "control_qr_url": "/control_qr.svg",
    })


@APP.get("/control_qr.svg")
async def control_qr_svg():
    control_url = get_control_url()

    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        fallback_svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="480" height="480" viewBox="0 0 480 480">
  <rect width="480" height="480" fill="#ffffff"/>
  <text x="240" y="185" text-anchor="middle" font-size="24" fill="#111827">QRコード生成には qrcode が必要です</text>
  <text x="240" y="230" text-anchor="middle" font-size="20" fill="#111827">pip install qrcode</text>
  <text x="240" y="285" text-anchor="middle" font-size="16" fill="#111827">{control_url}</text>
</svg>
"""
        return Response(
            content=fallback_svg.encode("utf-8"),
            media_type="image/svg+xml",
        )

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
        image_factory=qrcode.image.svg.SvgPathImage,
    )
    qr.add_data(control_url)
    qr.make(fit=True)

    img = qr.make_image()
    buf = BytesIO()
    img.save(buf)

    return Response(
        content=buf.getvalue(),
        media_type="image/svg+xml",
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


@APP.post("/api/birdnet/reset")
async def api_birdnet_reset():
    state = get_state()
    running = bool(state.get("birdnet", {}).get("running", False))

    update_state("birdnet", {
        "status": "running" if running else "ready",
        "capture_status": "ready",
        "analyze_status": "ready",
        "last_elapsed_seconds": None,
        "latest_detection": None,
    })

    return JSONResponse({
        "ok": True,
        "message": "birdnet reset",
    })


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


@APP.post("/api/mimic/reset")
async def api_mimic_reset():
    state = get_state()

    if bool(state.get("mimic", {}).get("running", False)):
        return JSONResponse(
            {"error": "鳴きまねチャレンジ実行中はリセットできません。"},
            status_code=409,
        )

    update_state("mimic", {
        "enabled": True,
        "running": False,
        "status": "ready",
        "active_species_key": None,
        "latest_score": None,
    })

    update_state("control", {
        "mimic_command": None,
        "active_species_key": None,
    })

    return JSONResponse({
        "ok": True,
        "message": "mimic reset",
    })


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

    if CONFIG is None:
        return JSONResponse(
            {"error": "Config is not initialized"},
            status_code=500,
        )

    if species_key not in CONFIG["mimic"]["species"]:
        return JSONResponse(
            {"error": f"Unknown species key: {species_key}"},
            status_code=400,
        )

    update_state("mimic", {
        "active_species_key": species_key,
    })
    update_state("control", {
        "active_species_key": species_key,
    })

    try:
        result = await asyncio.to_thread(
            MIMIC_SERVICE.record_and_score,
            species_key,
            reference_index,
        )

        update_state("mimic", {
            "active_species_key": species_key,
            "latest_score": result,
        })
        update_state("control", {
            "active_species_key": species_key,
        })

        return JSONResponse(result)

    except Exception as e:
        update_state("mimic", {
            "running": False,
            "status": f"mimic error: {e}",
            "active_species_key": species_key,
        })
        return JSONResponse({"error": str(e)}, status_code=500)


@APP.post("/api/control/mimic/{species_key}")
async def api_control_mimic(
    species_key: str,
    source: str = Query(default="control"),
):
    if CONFIG is None:
        return JSONResponse(
            {"error": "Config is not initialized"},
            status_code=500,
        )

    if species_key not in CONFIG["mimic"]["species"]:
        return JSONResponse(
            {"error": f"Unknown species key: {species_key}"},
            status_code=400,
        )

    command = make_mimic_command(species_key, source=source)

    update_state("control", {
        "mimic_command": command,
        "active_species_key": species_key,
    })
    update_state("mimic", {
        "active_species_key": species_key,
    })

    return JSONResponse({
        "ok": True,
        "message": "mimic command issued",
        "command": command,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        APP,
        host="0.0.0.0",
        port=8000,
    )