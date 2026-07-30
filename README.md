# DataMine AI — IBM AI Builders Challenge (Wildcard: Future of Work)

> A prompt-driven AI agent pipeline that automates end-to-end data mining and curation.  
> The user describes what they need training data for; the system collects, cleans,
> quality-filters, and deduplicates raw text and image data from public sources into a
> clean, training-ready dataset.

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Architecture](#architecture)
3. [Pipeline steps](#pipeline-steps)
4. [Tech stack](#tech-stack)
5. [Getting started](#getting-started)
6. [API reference](#api-reference)
7. [Estimated savings methodology](#estimated-savings-methodology)
8. [Data sources](#data-sources)
9. [Environment variables](#environment-variables)
10. [Project structure](#project-structure)

---

## Project overview

Preparing data to train an AI model is slow, manual, and wasteful.  
This project automates the entire workflow using an AI agent chain driven by a single
natural-language request from the user.

**The deliverable is the clean, ready-to-train dataset — this system does NOT train a
model itself.**

---

## Architecture

```
User prompt (free-form text)
        │
        ▼
┌─────────────────────┐
│  1. Intent parser   │  Groq LLM → extracts topic + 2 search keywords
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Collector       │  arXiv API (2 queries) + Openverse API
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Cleaner         │  Strip LaTeX, boilerplate, broken entries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Quality filter  │  Groq LLM (KEEP / DROP per entry)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. Deduplicator    │  sentence-transformers cosine sim (text)
│                     │  imagehash pHash (images)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Structurer      │  JSONL + image metadata + run report + ZIP
└─────────────────────┘
```

The FastAPI backend streams pipeline progress to the frontend via
**Server-Sent Events (SSE)**, so the UI updates in real time.

---

## Pipeline steps

| Step | Agent | Key technology |
|------|-------|----------------|
| 1 | Intent parser | Groq API — `llama-3.3-70b-versatile` |
| 2 | Collector | arXiv Atom API · Openverse REST API |
| 3 | Cleaner | Python `re`, `unicodedata` |
| 4 | Quality filter | Groq API — `llama-3.1-8b-instant` |
| 5 | Deduplicator | `sentence-transformers/all-MiniLM-L6-v2` (cosine sim ≥ 0.85) · `imagehash` pHash (Hamming ≤ 10) |
| 6 | Structurer | JSONL · JSON · ZIP · savings math |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14 · React 18 · Tailwind CSS |
| Backend | FastAPI · Python 3.11+ |
| AI / LLM | Groq API (Llama 3.3 70B / Llama 3.1 8B) |
| Text embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Image dedup | `imagehash` (pHash) · `Pillow` |
| Data sources | arXiv API · Openverse API |
| Deployment | Vercel (frontend) · Render / Railway (backend) |

---

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+
- A free [Groq API key](https://console.groq.com)

### Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY

uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install

cp .env.example .env.local
# NEXT_PUBLIC_API_URL is already set to http://localhost:8000

npm run dev
```

Open `http://localhost:3000`.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mine` | Start a run; returns `{ run_id }` |
| `GET` | `/mine/{run_id}/stream?prompt=…` | SSE stream of pipeline events |
| `GET` | `/mine/{run_id}/download` | Download the finished dataset ZIP |
| `GET` | `/health` | Health check |

### SSE event shape

```json
{
  "run_id": "uuid",
  "step":   "collect_done",
  "message": "Collected 45 text entries and 20 images.",
  "raw_text_count": 45,
  "raw_image_count": 20
}
```

The final `done` event carries the full `report` object (counts, savings, dup pairs,
sample previews).

---

## Estimated savings methodology

> **Label:** *Estimated savings, based on published NVIDIA A100 specifications and
> standard linear training-time scaling.*

The savings shown in the run dashboard are computed in this exact order so every
number is defensible rather than arbitrary:

### Step 1 — Data reduction %

```
data_reduction_pct = duplicates_removed ÷ total_raw_entries
```

This is a **directly measured** figure from the live run — not an estimate.

### Step 2 — GPU-hours saved

```
gpu_hours_saved = data_reduction_pct × BASELINE_GPU_HOURS
```

`BASELINE_GPU_HOURS = 8` is a labeled approximation of how long a typical
fine-tuning run on this dataset would take on a single A100.  
**Assumption:** training time scales roughly linearly with data volume for a fixed
model and epoch count. This is an approximation; real-world scaling depends on
batch size, optimizer, and hardware.

### Step 3 — Energy saved

```
energy_saved_kWh = gpu_hours_saved × (400 W ÷ 1000)
```

GPU power draw: **400 W** (max TDP for NVIDIA A100 SXM4).  
**Source:** *NVIDIA A100 Tensor Core GPU Datasheet*, SXM4 variant, 400 W TDP.
[https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)

### Step 4 — Cost saved

```
cost_saved_USD = gpu_hours_saved × $2.49 / GPU-hour
```

**Source:** Lambda Labs on-demand A100 SXM4 pricing page —
[https://lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
(rate pulled at project build time).

### Step 5 — Carbon and water saved

```
carbon_saved_kg_CO2 = energy_saved_kWh × 386 g/kWh ÷ 1000
water_saved_litres  = energy_saved_kWh × 1.8 L/kWh
```

**Carbon source:** US EPA eGRID 2022 national average — 386 g CO₂ / kWh.  
[https://www.epa.gov/egrid](https://www.epa.gov/egrid)

**Water source:** Patterson et al., *"Carbon Emissions and Large Neural Network
Training"* (2021) — ~1.8 L / kWh data-center WUE.  
[https://arxiv.org/abs/2104.10350](https://arxiv.org/abs/2104.10350)

---

## Data sources

### arXiv (text)
- **API:** `https://export.arxiv.org/api/query` (Atom/XML, no auth required)
- Two queries per run (one per extracted keyword) to surface overlapping papers
  that the dedup agent can catch
- Covers physics, environmental science, ecology, biology, and most other academic
  fields
- Run: `search_query=all:<keyword>&max_results=30`

### Openverse (images)
- **API:** `https://api.openverse.org/v1/images/` (JSON REST)
- Aggregates openly-licensed images from museums, NASA, Flickr Commons, and
  hundreds of other institutions
- Every returned image carries a verified CC license
- Anonymous requests are rate-limited; supply `OPENVERSE_CLIENT_ID` +
  `OPENVERSE_CLIENT_SECRET` for higher limits (free registration)

---

## Environment variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key (free at console.groq.com) |
| `OPENVERSE_CLIENT_ID` | ✗ | Openverse OAuth client ID (higher rate limits) |
| `OPENVERSE_CLIENT_SECRET` | ✗ | Openverse OAuth client secret |
| `OUTPUT_DIR` | ✗ | Directory for run outputs (default: `outputs/`) |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | ✅ | Backend base URL (default: `http://localhost:8000`) |

---

## Project structure

```
project-pipeline/
├── backend/
│   ├── main.py                  # FastAPI app, SSE orchestration
│   ├── requirements.txt
│   ├── .env.example
│   └── agents/
│       ├── __init__.py
│       ├── intent_parser.py     # Groq → topic + keywords
│       ├── collector.py         # arXiv + Openverse
│       ├── cleaner.py           # text & image cleaning
│       ├── quality_filter.py    # Groq KEEP/DROP judge
│       ├── deduplicator.py      # cosine sim + pHash
│       └── structurer.py        # JSONL + ZIP + savings
└── frontend/
    ├── package.json
    ├── next.config.js
    ├── tailwind.config.ts
    ├── tsconfig.json
    ├── .env.example
    └── src/
        └── app/
            ├── layout.tsx
            ├── globals.css
            └── page.tsx         # Single-page UI
```

---

## Team

| Role | Focus |
|------|-------|
| Frontend | Next.js UI, live status log, dashboard |
| Backend / orchestration | FastAPI, agent chain wiring |
| AI / data layer | Embeddings, dedup logic, Groq integration |
| Data-source integration | arXiv + Openverse API handling, cleaning |
| Demo / polish + video | Report formatting, savings math, 3-minute video |
