"""
The transport layer: FastAPI + one WebSocket per voice session.

Wire protocol (deliberately minimal, mirrors what commercial voice APIs do):

  client -> server   binary frames: 16-bit LE PCM, 16 kHz mono mic audio
  server -> client   binary frames: 16-bit LE PCM TTS audio (rate in tts_start)
                     text frames:   JSON events (ready/state/transcript/reply/
                                    tts_start/tts_sentence/interrupted/...)

Serving decisions worth noticing:
  - Models load ONCE in the lifespan hook and are shared by all sessions;
    per-session state (VAD RNN state, endpointer, orchestrator) is created
    per connection. Weights are immutable => safe to share; state is not.
  - GET /healthz answers only after models are loaded (a load balancer must
    not route traffic to a replica still downloading weights).
  - GET /metrics exposes the latency summary — the numbers to watch while
    tuning end_silence_ms / model sizes.

Run:  uv run python -m server.app   then open http://127.0.0.1:8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline.config import PipelineConfig
from src.pipeline.metrics import Metrics
from src.pipeline.orchestrator import VoiceSession

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"

cfg = PipelineConfig()
metrics = Metrics()
engines: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import + construct here so `--help`/tests don't pay the model-load cost.
    from src.pipeline.engines import make_asr, make_tts

    print(f"engines: vad={cfg.vad_engine} asr={cfg.asr_engine} tts={cfg.tts_engine}")
    engines["asr"] = make_asr(cfg)
    engines["tts"] = make_tts(cfg)
    yield
    engines.clear()


app = FastAPI(title="voice-pipeline", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    ok = "asr" in engines and "tts" in engines
    return JSONResponse({"status": "ok" if ok else "loading"}, status_code=200 if ok else 503)


@app.get("/metrics")
async def get_metrics():
    return metrics.summary()


@app.get("/")
async def index():
    return FileResponse(CLIENT_DIR / "index.html")


app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")


@app.websocket("/ws")
async def ws_session(ws: WebSocket):
    await ws.accept()

    # Per-session objects: the VAD carries context/RNN state across windows,
    # the endpointer carries turn state, and the responder carries the
    # conversation history — none may be shared between users.
    from src.pipeline.engines import make_responder, make_vad

    session = VoiceSession(
        vad=make_vad(cfg),
        asr=engines["asr"],
        tts=engines["tts"],
        responder=make_responder(cfg),
        cfg=cfg,
        metrics=metrics,
        emit_json=ws.send_json,
        emit_audio=ws.send_bytes,
    )
    await session.start()
    try:
        while True:
            frame = await ws.receive()
            if frame.get("bytes"):
                await session.feed(frame["bytes"])
            elif frame.get("text") is not None:
                pass  # no client->server JSON commands yet
            elif frame.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port)
