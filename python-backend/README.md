# OpenMAIC Python Backend

A FastAPI-based Python rewrite of the OpenMAIC Next.js API layer.  
Implements all 25 original API routes with a clean, extensible architecture.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Web framework | FastAPI + uvicorn |
| LLM (OpenAI/compatible) | `openai` Python SDK |
| LLM (Anthropic) | `anthropic` SDK |
| LLM (Google) | `google-generativeai` SDK |
| Config | pydantic-settings + YAML |
| Async HTTP | httpx |
| PDF parsing | pypdf (built-in) + MinerU API |
| Storage | File-based JSON (swap to DB easily) |

## Project Structure

```
python-backend/
├── app/
│   ├── main.py              # FastAPI app, CORS, router mounting
│   ├── config.py            # Settings (env + server-providers.yml)
│   └── api/                 # All 25 route handlers
│       ├── health.py
│       ├── classroom.py
│       ├── generate_classroom.py
│       ├── chat.py
│       ├── pbl_chat.py
│       ├── tts.py
│       ├── transcription.py
│       ├── azure_voices.py
│       ├── parse_pdf.py
│       ├── web_search.py
│       ├── quiz_grade.py
│       ├── proxy_media.py
│       ├── classroom_media.py
│       ├── server_providers.py
│       ├── verify_model.py
│       ├── verify_pdf_provider.py
│       ├── verify_image_provider.py
│       ├── verify_video_provider.py
│       └── generate/
│           ├── scene_outlines_stream.py
│           ├── scene_content.py
│           ├── scene_actions.py
│           ├── agent_profiles.py
│           ├── image.py
│           └── video.py
├── core/
│   ├── providers/           # LLM / TTS / ASR / Image / Video / PDF factories
│   ├── generation/          # Classroom generation pipeline
│   ├── storage/             # Classroom & job persistence
│   ├── security/            # SSRF guard
│   └── web_search/          # Tavily integration
├── models/                  # Pydantic data models
├── server-providers.yml     # Server-side API keys (optional)
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
cd python-backend

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — add at least one LLM API key

# 3. Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs.

## API Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check + capabilities |
| GET | `/api/server-providers` | List configured providers |
| POST | `/api/classroom` | Save a classroom |
| GET | `/api/classroom?id=...` | Load a classroom |
| POST | `/api/generate-classroom` | Submit generation job |
| GET | `/api/generate-classroom/{jobId}` | Poll job status |
| POST | `/api/chat` | SSE chat stream |
| POST | `/api/pbl/chat` | PBL agent chat |
| POST | `/api/generate/tts` | Text-to-speech |
| POST | `/api/transcription` | Speech-to-text |
| POST | `/api/azure-voices` | List Azure TTS voices |
| POST | `/api/parse-pdf` | Parse PDF document |
| POST | `/api/web-search` | Tavily web search |
| POST | `/api/quiz-grade` | Grade quiz answer with LLM |
| POST | `/api/proxy-media` | CORS proxy for remote media |
| GET | `/api/classroom-media/{id}/{path}` | Serve stored media files |
| POST | `/api/verify-model` | Test LLM connectivity |
| POST | `/api/verify-pdf-provider` | Test PDF provider |
| POST | `/api/verify-image-provider` | Test image provider |
| POST | `/api/verify-video-provider` | Test video provider |
| POST | `/api/generate/scene-outlines-stream` | SSE outline streaming |
| POST | `/api/generate/scene-content` | Generate scene content |
| POST | `/api/generate/scene-actions` | Generate scene actions |
| POST | `/api/generate/agent-profiles` | Generate agent profiles |
| POST | `/api/generate/image` | Generate image |
| POST | `/api/generate/video` | Generate video |

## Configuration

### Option 1: `.env` file
```env
DEFAULT_MODEL=openai:gpt-4o-mini
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### Option 2: `server-providers.yml`
```yaml
providers:
  openai:
    apiKey: sk-...
  deepseek:
    apiKey: sk-...
    baseUrl: https://api.deepseek.com/v1
```

### Supported LLM model strings
- `openai:gpt-4o-mini`
- `anthropic:claude-3-5-haiku-20241022`
- `google:gemini-2.0-flash`
- `deepseek:deepseek-chat`
- `qwen:qwen-plus`
- `glm:glm-4-flash`

## Integrating with the Frontend

Point the Next.js frontend to this backend by setting:
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```
(or configure a reverse proxy so `/api/*` routes to this service)

## Extending

**Add a new LLM provider:**  
Add its base URL to `_OPENAI_COMPATIBLE_BASE_URLS` in `core/providers/llm.py` and the env/YAML key in `app/config.py`.

**Add a new TTS provider:**  
1. Add a case to `generate_tts()` in `core/providers/tts.py`  
2. Add config keys in `app/config.py`

**Swap storage backend (e.g. to PostgreSQL):**  
Rewrite `core/storage/classroom_store.py` and `core/storage/job_store.py` — the API layer is unchanged.
