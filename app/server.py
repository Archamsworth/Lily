"""FastAPI server – the central hub for Lily.

Serves the web frontend and provides:
  - REST  POST /api/chat        – text-input chat
  - REST  POST /api/voice       – audio file upload → STT → chat
  - WS    /ws                   – real-time streaming (preferred)
  - POST  /api/wake_word/start  – enable microphone wake-word listener
  - POST  /api/wake_word/stop   – disable wake-word listener
  - GET   /api/status           – system readiness check
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.avatar.expressions import ExpressionEvent, TextToken, parse_stream
from app.config import load_config
from app.llm.model import LilyLLM
from app.llm.rag import RAGRetriever
from app.stt.whisper_stt import WhisperSTT
from app.tts.piper_tts import PiperTTS
from app.wake_word.detector import WakeWordDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(title="Lily", description="Virtual companion API", version="0.1.0")

_cfg: dict[str, Any] = {}
_llm: LilyLLM | None = None
_rag: RAGRetriever | None = None
_stt: WhisperSTT | None = None
_tts: PiperTTS | None = None
_wake: WakeWordDetector | None = None

# Active WebSocket connections (for wake-word push notifications)
_ws_connections: set[WebSocket] = set()


@app.on_event("startup")
async def _startup() -> None:
    global _cfg, _llm, _rag, _stt, _tts, _wake
    try:
        _cfg = load_config("config.yaml")
    except FileNotFoundError:
        logger.warning("config.yaml not found – using built-in defaults.")
        _cfg = {"lily": {}}

    lily = _cfg["lily"]

    try:
        _llm = LilyLLM(lily.get("llm", {}))
        logger.info("LLM loaded.")
    except Exception as exc:
        logger.error("Failed to load LLM: %s", exc)

    _rag = RAGRetriever(lily.get("rag", {}))

    try:
        _stt = WhisperSTT(lily.get("stt", {}))
        logger.info("Whisper STT loaded.")
    except Exception as exc:
        logger.error("Failed to load Whisper: %s", exc)

    try:
        _tts = PiperTTS(lily.get("tts", {}))
        logger.info("Piper TTS ready.")
    except Exception as exc:
        logger.error("Failed to init Piper TTS: %s", exc)

    if lily.get("wake_word", {}).get("enabled", True):
        try:
            _wake = WakeWordDetector(
                lily.get("wake_word", {}),
                on_detected=_on_wake_word,
            )
            logger.info("Wake-word detector initialised.")
        except Exception as exc:
            logger.error("Failed to init wake-word detector: %s", exc)


# Serve static files
_STATIC_DIR = Path(__file__).parent.parent / "web" / "static"
_ASSETS_DIR = Path(__file__).parent.parent / "web" / "assets"
_WEB_DIR = Path(__file__).parent.parent / "web"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")


# ---------------------------------------------------------------------------
# Wake-word callback
# ---------------------------------------------------------------------------

def _on_wake_word() -> None:
    """Called from the wake-word background thread when the keyword is heard."""
    asyncio.get_event_loop().call_soon_threadsafe(
        asyncio.ensure_future,
        _broadcast({"type": "wake_word", "detected": True}),
    )


async def _broadcast(message: dict) -> None:
    dead: set[WebSocket] = set()
    for ws in list(_ws_connections):
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _ws_connections.difference_update(dead)


# ---------------------------------------------------------------------------
# WebSocket – streaming endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _ws_connections.add(websocket)
    history: list[dict[str, str]] = []

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "text":
                await _handle_text_ws(websocket, msg.get("content", ""), history)

            elif msg_type == "audio":
                await _handle_audio_ws(websocket, msg.get("data", ""), history)

            elif msg_type == "clear_history":
                history.clear()
                await websocket.send_json({"type": "history_cleared"})

    except WebSocketDisconnect:
        pass
    finally:
        _ws_connections.discard(websocket)


async def _handle_text_ws(
    ws: WebSocket,
    user_text: str,
    history: list[dict[str, str]],
) -> None:
    if not user_text.strip():
        return
    if _llm is None:
        await ws.send_json({"type": "error", "message": "LLM not loaded"})
        return

    context = ""
    if _rag and RAGRetriever.should_search(user_text):
        context = await asyncio.to_thread(_rag.get_context, user_text)

    spoken_buffer: list[str] = []
    full_response: list[str] = []

    def _token_gen():
        yield from _llm.chat_stream(user_text, context=context, history=history)

    for item in parse_stream(_token_gen()):
        if isinstance(item, TextToken):
            await ws.send_json({"type": "token", "text": item.text})
            spoken_buffer.append(item.text)
            full_response.append(item.text)
        elif isinstance(item, ExpressionEvent):
            await ws.send_json({
                "type": "expression",
                "name": item.vrm_expression,
                "raw": item.raw,
                "description": item.description,
            })
            full_response.append(f"*{item.raw}*")

    spoken_text = "".join(spoken_buffer).strip()
    assistant_full = "".join(full_response).strip()

    # Update history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_full})

    # TTS
    if _tts and spoken_text:
        try:
            audio_bytes = await asyncio.to_thread(_tts.synthesize, spoken_text)
            if audio_bytes:
                await ws.send_json({
                    "type": "tts_audio",
                    "data": base64.b64encode(audio_bytes).decode(),
                    "mime": "audio/wav",
                })
        except Exception as exc:
            logger.error("TTS error: %s", exc)

    await ws.send_json({"type": "done"})


async def _handle_audio_ws(
    ws: WebSocket,
    audio_b64: str,
    history: list[dict[str, str]],
) -> None:
    if _stt is None:
        await ws.send_json({"type": "error", "message": "STT not loaded"})
        return
    try:
        audio_bytes = base64.b64decode(audio_b64)
        user_text = await asyncio.to_thread(_stt.transcribe_bytes, audio_bytes)
        await ws.send_json({"type": "stt_result", "text": user_text})
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"STT error: {exc}"})
        return

    if user_text:
        await _handle_text_ws(ws, user_text, history)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat_endpoint(payload: dict[str, Any]) -> JSONResponse:
    """Single-turn text chat (non-streaming)."""
    user_text: str = payload.get("message", "")
    if not user_text.strip():
        raise HTTPException(status_code=400, detail="'message' field is required")
    if _llm is None:
        raise HTTPException(status_code=503, detail="LLM not loaded")

    context = ""
    if _rag and RAGRetriever.should_search(user_text):
        context = await asyncio.to_thread(_rag.get_context, user_text)

    full = await asyncio.to_thread(_llm.chat, user_text, context)
    spoken = LilyLLM.strip_expressions(full)

    tts_audio_b64 = None
    if _tts and spoken:
        try:
            wav = await asyncio.to_thread(_tts.synthesize, spoken)
            tts_audio_b64 = base64.b64encode(wav).decode()
        except Exception as exc:
            logger.error("TTS error: %s", exc)

    return JSONResponse({
        "response": full,
        "spoken": spoken,
        "tts_audio": tts_audio_b64,
    })


@app.post("/api/voice")
async def voice_endpoint(audio: UploadFile = File(...)) -> JSONResponse:
    """Upload an audio file, transcribe it, then chat."""
    if _stt is None:
        raise HTTPException(status_code=503, detail="STT not loaded")

    audio_bytes = await audio.read()
    user_text = await asyncio.to_thread(_stt.transcribe_bytes, audio_bytes)

    if not user_text:
        raise HTTPException(status_code=422, detail="Could not transcribe audio")

    if _llm is None:
        return JSONResponse({"stt": user_text, "response": None})

    context = ""
    if _rag and RAGRetriever.should_search(user_text):
        context = await asyncio.to_thread(_rag.get_context, user_text)

    full = await asyncio.to_thread(_llm.chat, user_text, context)
    spoken = LilyLLM.strip_expressions(full)

    tts_audio_b64 = None
    if _tts and spoken:
        try:
            wav = await asyncio.to_thread(_tts.synthesize, spoken)
            tts_audio_b64 = base64.b64encode(wav).decode()
        except Exception as exc:
            logger.error("TTS error: %s", exc)

    return JSONResponse({
        "stt": user_text,
        "response": full,
        "spoken": spoken,
        "tts_audio": tts_audio_b64,
    })


@app.post("/api/wake_word/start")
async def start_wake_word() -> JSONResponse:
    if _wake is None:
        raise HTTPException(status_code=503, detail="Wake-word detector not initialised")
    _wake.start()
    return JSONResponse({"status": "started"})


@app.post("/api/wake_word/stop")
async def stop_wake_word() -> JSONResponse:
    if _wake is None:
        raise HTTPException(status_code=503, detail="Wake-word detector not initialised")
    _wake.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "llm": _llm is not None,
        "stt": _stt is not None,
        "tts": _tts is not None,
        "wake_word": _wake is not None,
        "wake_word_running": _wake.is_running if _wake else False,
    })


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index() -> FileResponse:
    index = _WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index))
