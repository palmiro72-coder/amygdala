# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AMYGDALA is a neuromorphic AI orchestration layer that routes inference requests to optimal backends (Llama 70B, DeepSeek V3, Claude Sonnet) based on task complexity, urgency, and resource availability. Written in Python 3.13 using FastAPI.

## Commands

```bash
# Run CLI interface
python amygdala_core.py

# Run web server (default: 0.0.0.0:8000)
python server.py

# Run tests
python test_amygdala.py              # Unit tests only
python test_amygdala.py --live       # Include live API calls
python test_amygdala.py --benchmark  # Include performance benchmarks

# Install dependencies
pip install -r requirements-server.txt
```

## Architecture

The system uses a **neuromorphic metaphor** where each component maps to a brain structure:

**Three-tier routing** via `ThalamusRouter` in `amygdala_core.py`:
- **LIMBIC** (score 0.0–0.3): Simple queries → Llama 70B via Hyperbolic API
- **CORTICAL** (score 0.3–0.7): Moderate reasoning → DeepSeek V3 via Hyperbolic API
- **PREFRONTAL** (score 0.7–1.0): Complex analysis → Claude Sonnet via Anthropic API

Complexity scoring uses keyword analysis, word count, domain detection (medical queries get +0.3 boost), and neurotransmitter signal modifiers (e.g., NOREPINEPHRINE +0.3 for urgency keywords).

**Memory system** (`HippocampusMemory`): Working memory (last 10 interactions) + long-term storage with SHA256-based retrieval and emotional decay (0.95 rate).

**Inference backends** extend `InferenceBackend` ABC:
- `HyperbolicBackend`: Uses AsyncOpenAI client pointed at Hyperbolic endpoint
- `ClaudeBackend`: Uses httpx.AsyncClient for direct Anthropic API calls

**Web server** (`server.py`): FastAPI app with REST endpoints (`/api/process`, `/api/analyze`, `/api/stats`, `/health`) and WebSocket (`/ws`) for real-time neural state visualization. Serves `AMYGDALA_FINAL.html` as the web UI.

## Key Files

- `amygdala_core.py` — Core engine: config, routing, memory, backends, orchestrator, CLI
- `server.py` — FastAPI web server with REST + WebSocket endpoints
- `test_amygdala.py` — Test suite with unit tests, live tests, and benchmarks
- `AMYGDALA_FINAL.html` — Interactive neural visualization web interface
- `install.sh` — Production deployment script (systemd service, supports apt/yum)

## Configuration

All config is via environment variables in `.env` (copy from `.env.example`). Key vars:
- `ANTHROPIC_API_KEY`, `HYPERBOLIC_API_KEY` — Required API keys
- `AMYGDALA_HOST`, `AMYGDALA_PORT` — Server binding (default: 0.0.0.0:8000)
- `AMYGDALA_THRESHOLD_LOW`, `AMYGDALA_THRESHOLD_HIGH` — Routing thresholds (default: 0.3, 0.7)
- `AMYGDALA_CLAUDE_MODEL`, `AMYGDALA_HYPERBOLIC_FAST`, `AMYGDALA_HYPERBOLIC_REASONING` — Model overrides

## Conventions

- All processing is async (`async/await` throughout)
- Pydantic models for API request/response validation
- Dataclasses for internal configuration (`AmygdalaConfig`)
- Enums for controlled values (`TaskComplexity`, `NeurotransmitterSignal`)
- Medical domain queries are always routed to higher complexity tiers for safety
