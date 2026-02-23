# Lily – Virtual Companion

Lily is a locally-running AI companion featuring:

| Feature | Implementation |
|---|---|
| 3-D avatar | VRM file rendered with Three.js + @pixiv/three-vrm |
| Language model | [qwen2.5-3b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) via **llama-cpp-python** |
| Web search (RAG) | DuckDuckGo search + BeautifulSoup page scraping |
| Speech-to-text | **OpenAI Whisper** (runs fully offline) |
| Text-to-speech | **Piper TTS** (fast, offline neural TTS) |
| Wake word | **openwakeword** (offline, "Hey Jarvis" built-in; custom model supported) |
| Button trigger | Hold-to-talk mic button in the web UI |
| Expression streaming | `*action*` markers parsed from LLM output → VRM blend-shape animation |

---

## Project layout

```
Lily/
├── main.py                  # Entry point (uvicorn launcher)
├── config.yaml              # All runtime settings
├── requirements.txt
├── app/
│   ├── config.py            # Config loader (supports env-var overrides)
│   ├── server.py            # FastAPI app (REST + WebSocket)
│   ├── llm/
│   │   ├── model.py         # LilyLLM – llama-cpp wrapper, streaming chat
│   │   └── rag.py           # RAGRetriever – DuckDuckGo + page scraping
│   ├── stt/
│   │   └── whisper_stt.py   # WhisperSTT – file or raw-PCM transcription
│   ├── tts/
│   │   └── piper_tts.py     # PiperTTS – subprocess wrapper for piper
│   ├── wake_word/
│   │   └── detector.py      # WakeWordDetector – openwakeword listener
│   └── avatar/
│       └── expressions.py   # Expression parser (stream *action* markers)
├── web/
│   ├── index.html           # Single-page frontend
│   ├── assets/              # Drop avatar.vrm here
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── avatar.js    # Three.js + VRM rendering + idle blink
│           └── app.js       # WebSocket client, mic, TTS playback
└── tests/                   # pytest test suite (all external deps mocked)
```

---

## Quick-start

### 1. Prerequisites

- Python 3.10+ (3.12 recommended)
- A C++ compiler (for `llama-cpp-python`)
- `piper` binary on your `PATH` – install with `pip install piper-tts`
- PortAudio (for PyAudio / sounddevice): `sudo apt install portaudio19-dev`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration** (optional): install `llama-cpp-python` with CUDA support:
> ```bash
> CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
> ```
> Then set `n_gpu_layers` in `config.yaml` to the number of layers to offload.

### 3. Place the language model

```
mkdir -p models
cp /path/to/qwen2.5-3b-instruct-q4_k_m.gguf models/
```

The default path in `config.yaml` is `models/qwen2.5-3b-instruct-q4_k_m.gguf`.

### 4. Download a Piper voice

```bash
mkdir -p voices
# example – en_US lessac medium quality
wget -P voices https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -P voices https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Update `tts.voice` in `config.yaml` if you choose a different voice.

### 5. Add your VRM avatar (optional)

Copy any VRM 0.x or 1.0 file to `web/assets/avatar.vrm`.
If omitted Lily shows a simple capsule placeholder and setup still works.

### 6. Run

```bash
python main.py
```

Open **http://127.0.0.1:8000** in your browser.

---

## Configuration

All settings live in `config.yaml`. The most important ones:

| Key | Default | Description |
|---|---|---|
| `lily.llm.model_path` | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | Path to GGUF model |
| `lily.llm.n_gpu_layers` | `0` | Layers to offload to GPU |
| `lily.stt.model` | `base` | Whisper model size (`tiny` → `large`) |
| `lily.tts.voice` | `en_US-lessac-medium` | Piper voice name (stem of .onnx file) |
| `lily.wake_word.model` | `hey_jarvis` | openwakeword model name or path |
| `lily.wake_word.threshold` | `0.5` | Detection sensitivity (0–1) |
| `lily.rag.enabled` | `true` | Enable web search augmentation |
| `lily.server.port` | `8000` | HTTP server port |

Environment variables override config values:

| Variable | Config key |
|---|---|
| `LILY_MODEL_PATH` | `lily.llm.model_path` |
| `LILY_HOST` | `lily.server.host` |
| `LILY_PORT` | `lily.server.port` |
| `LILY_WHISPER_MODEL` | `lily.stt.model` |
| `LILY_PIPER_VOICE` | `lily.tts.voice` |

---

## API reference

### REST

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web frontend |
| `POST` | `/api/chat` | `{"message": "..."}` → `{response, spoken, tts_audio}` |
| `POST` | `/api/voice` | Multipart `audio` file → STT + chat + TTS |
| `POST` | `/api/wake_word/start` | Start microphone wake-word listener |
| `POST` | `/api/wake_word/stop` | Stop wake-word listener |
| `GET` | `/api/status` | `{llm, stt, tts, wake_word, wake_word_running}` |

### WebSocket `/ws`

**Client → Server**

```jsonc
{"type": "text",          "content": "Hello Lily!"}
{"type": "audio",         "data": "<base64 audio>"}
{"type": "clear_history"}
```

**Server → Client**

```jsonc
{"type": "token",       "text": "..."}               // streaming LLM token
{"type": "expression",  "name": "happy", "raw": "smiles warmly", "description": "..."}
{"type": "tts_audio",   "data": "<base64 WAV>",  "mime": "audio/wav"}
{"type": "stt_result",  "text": "transcribed text"}
{"type": "wake_word",   "detected": true}
{"type": "done"}
{"type": "error",       "message": "..."}
```

---

## Expression system

The LLM is prompted to embed `*action*` markers inline with its replies:

```
*smiles warmly* Sure, let me help! *tilts head curiously* What would you like to know?
```

The backend (`app/avatar/expressions.py`) parses these markers in real-time as tokens stream from the model. Each marker is:

1. **Mapped** to a VRM blend-shape name (happy, sad, angry, surprised, relaxed, neutral)
2. **Sent** to the frontend as an `expression` WebSocket event
3. **Stripped** from the text sent to Piper TTS so it is not spoken aloud

The frontend (`avatar.js`) calls `applyExpression(name, intensity, duration)` which sets the VRM blend-shape and fades it back to neutral after `duration` milliseconds.

---

## Running tests

```bash
pytest tests/ -v
```

All tests use mocks – no model files or hardware are required.

---

## Roadmap (future upgrades)

- [ ] Custom "Hey Lily" openwakeword model training
- [ ] Micro / macro expression animation driven by an animation AI
- [ ] Lip-sync (VRM viseme shapes: aa, ih, ou, ee, oh) driven by TTS audio
- [ ] Mobile-friendly overlay mode (similar to MateXengine)
- [ ] Packaged desktop app (Electron / Tauri)
- [ ] Plugin system for additional skills / tools
