# DataMine AI — IBM AI Builders Challenge (Wildcard: Future of Work)

> A prompt-driven AI agent pipeline that automates end-to-end data mining and curation.  
> The user describes what they need training data for; the system collects, cleans,
> quality-filters, deduplicates, and structures raw text and image data from public
> sources into a clean, Alpaca-format training-ready dataset.

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

The pipeline collects academic papers and openly-licensed images, cleans and filters
them with LLM-based quality judgements, removes near-duplicates, then synthesises the
surviving text into **Alpaca-format instruction-tuning records** ready for LLM fine-tuning.

**The deliverable is the clean, ready-to-train dataset — this system does NOT train a
model itself.**

---

## Architecture

```
User prompt (free-form text)
        │
        ▼
┌─────────────────────┐
│  1. Intent parser   │  Groq LLM (Llama 3.3 70B) → topic + 3 search queries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Collector       │  Wikipedia · Semantic Scholar · DuckDuckGo (text)
│                     │  Openverse · Wikimedia Commons · DDG Images (images)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Cleaner         │  Strip HTML, LaTeX, citations; Unicode normalise;
│                     │  drop boilerplate / too-short entries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Quality filter  │  Groq LLM (Llama 3.1 8B) — batch YES/NO specificity
│                     │  gate; heuristic pre-filter for images
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  5. Deduplicator    │  HF Inference API cosine similarity (text, ≥ 0.78)
│                     │  imagehash pHash (images, Hamming ≤ 8)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  6. Structurer      │  Groq LLM (Llama 3.1 8B) → Alpaca JSONL
│                     │  + image metadata JSON + run report + ZIP
└─────────────────────┘
```

The FastAPI backend streams pipeline progress to the frontend via
**Server-Sent Events (SSE)**, so the UI updates in real time as each step completes.

---

## Pipeline steps

| Step | Agent | What it does | Key technology |
|------|-------|-------------|----------------|
| 1 | Intent parser | Extracts a topic label + 3 complementary search queries from the user's free-form prompt | Groq API — `llama-3.3-70b-versatile` |
| 2 | Collector | Fetches text from Wikipedia, Semantic Scholar (2 queries per intent query), and DuckDuckGo (fallback). Fetches images from Openverse, Wikimedia Commons, and DDG Images | Wikipedia REST API · Semantic Scholar API · DuckDuckGo · Openverse API · Wikimedia Commons API |
| 3 | Cleaner | Strips HTML, LaTeX math, citation brackets; Unicode NFKC normalisation; drops entries shorter than 80 chars or matching boilerplate patterns | Python `re`, `unicodedata` |
| 4 | Quality filter | Hard keyword gate, then batch Groq YES/NO specificity judgement (20 text / 30 image entries per call); heuristic domain/URL/license pre-filter for images | Groq API — `llama-3.1-8b-instant` |
| 5 | Deduplicator | Embeds abstracts via HF Inference API; drops cross-document pairs with cosine similarity ≥ 0.78. Images deduplicated by pHash with Hamming distance ≤ 8 | `sentence-transformers/all-MiniLM-L6-v2` (HF Inference API) · `imagehash` · `Pillow` |
| 6 | Structurer | Converts surviving text entries into Alpaca `{instruction, input, output}` records via Groq (batches of 12); writes `texts.jsonl`, `images_metadata.json`, `run_report.json`, and a ZIP archive | Groq API — `llama-3.1-8b-instant` · Python `zipfile` |

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14 · React 18 · Tailwind CSS |
| Backend | FastAPI · Python 3.11+ · Server-Sent Events |
| AI / LLM | Groq API — `llama-3.3-70b-versatile` (intent parsing) · `llama-3.1-8b-instant` (quality filter + Alpaca synthesis) |
| Text embeddings | HF Inference API — `sentence-transformers/all-MiniLM-L6-v2` (no local PyTorch required) |
| Image dedup | `imagehash` (pHash) · `Pillow` |
| Data sources — text | Semantic Scholar API · Wikipedia REST API · DuckDuckGo (fallback) |
| Data sources — images | Openverse API · Wikimedia Commons API · DuckDuckGo Images (fallback) |
| Deployment | Vercel (frontend) · Render (backend) |

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
sample Alpaca record previews).

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
baseline_gpu_hours = max(0.1,  raw_entry_count / 100)
gpu_hours_saved    = data_reduction_pct × baseline_gpu_hours
```

Baseline GPU hours scale with the actual raw entry count for the run
(1 GPU-hour per 100 raw entries on a single A100).  
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

**Source:** Lambda Labs on-demand A100 SXM4 pricing —  
[https://lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)

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

### Text

| Source | API | Notes |
|--------|-----|-------|
| **Semantic Scholar** | `https://api.semanticscholar.org/graph/v1/paper/search` | Primary academic source — 2 complementary queries per intent query; no auth required for basic access |
| **Wikipedia** | `https://en.wikipedia.org/api/rest_v1/page/summary/` | REST summary API; run for every intent query |
| **DuckDuckGo** | DuckDuckGo search (HTML scraping) | Fallback supplement when primary sources return fewer than 10 entries per query |

### Images

| Source | API | Notes |
|--------|-----|-------|
| **Openverse** | `https://api.openverse.org/v1/images/` | Aggregates openly-licensed images from museums, NASA, Flickr Commons and hundreds of institutions; every image has a verified CC license |
| **Wikimedia Commons** | `https://commons.wikimedia.org/w/api.php` | Free-license media from the Wikimedia ecosystem |
| **DuckDuckGo Images** | DDG image search | Fallback supplement when Openverse + Wikimedia return fewer than 10 images |

---

## Environment variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key (free at console.groq.com) |
| `OPENVERSE_CLIENT_ID` | ✗ | Openverse OAuth client ID (higher rate limits) |
| `OPENVERSE_CLIENT_SECRET` | ✗ | Openverse OAuth client secret |
| `HF_API_TOKEN` | ✗ | Hugging Face Inference API token (higher embedding rate limits) |
| `OUTPUT_DIR` | ✗ | Directory for run outputs (default: `outputs/`; use `/tmp/outputs` on Render) |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | ✅ | Backend base URL (default: `http://localhost:8000`) |

---

## Project structure

```
project-pipeline/
├── render.yaml                  # Render Blueprint for backend deployment
├── backend/
│   ├── main.py                  # FastAPI app + SSE pipeline orchestration
│   ├── requirements.txt
│   ├── .env.example
│   └── agents/
│       ├── intent_parser.py     # Groq → topic + 3 search queries
│       ├── collector.py         # Wikipedia + Semantic Scholar + DDG (text)
│       │                        # Openverse + Wikimedia + DDG Images (images)
│       ├── cleaner.py           # HTML/LaTeX/citation stripping, normalisation
│       ├── quality_filter.py    # Groq batch YES/NO relevance judge + heuristics
│       ├── deduplicator.py      # HF cosine similarity (text) + pHash (images)
│       └── structurer.py        # Alpaca JSONL synthesis + ZIP + savings report
└── frontend/
    ├── package.json
    ├── next.config.js
    ├── tailwind.config.ts
    ├── tsconfig.json
    ├── .env.example
    └── src/
        └── app/
            ├── layout.tsx       # Root layout + metadata
            ├── globals.css
            └── page.tsx         # Single-page UI with live SSE log + dashboard
```

---

## Output format

Each pipeline run produces a ZIP archive containing:

| File | Format | Contents |
|------|--------|----------|
| `texts.jsonl` | JSONL | Alpaca records: `{instruction, input, output, _source, _search_term}` |
| `images_metadata.json` | JSON | Array of image entries with title, URL, thumbnail, license, source |
| `run_report.json` | JSON | Full run report: counts, savings, duplicate pairs, sample previews |

---

## Team

| Role | Focus |
|------|-------|
| Frontend | Next.js UI, live status log, dashboard |
| Backend / orchestration | FastAPI, SSE streaming, agent chain wiring |
| AI / data layer | Groq integration, Alpaca synthesis, quality filtering |
| Data-source integration | Semantic Scholar + Wikipedia + Openverse + Wikimedia handling, cleaning |
| Demo / polish | Report formatting, savings methodology, presentation |
