# Project Context — IBM AI Builders Challenge (Wildcard: Future of Work)

## One-line pitch
A prompt-driven AI agent pipeline that automates end-to-end data mining and curation — a user describes in plain language what they need training data for, and the system collects, cleans, quality-filters, and deduplicates raw text and image data from public sources into a clean, training-ready dataset — eliminating the manual, repetitive work of preparing data for AI training.

## Problem statement
Preparing data to train an AI model is slow, manual, and wasteful. Teams spend enormous time collecting raw data from scattered sources, cleaning it, and removing redundant/duplicate content — and skipping this step means training on bloated, duplicate-heavy datasets, which wastes real compute, energy, and money. This project automates that entire workflow using AI agents, driven by a single natural-language request from the user.

## Scope clarification (important, read before building)
The deliverable is the **clean, ready-to-train dataset** — the system does NOT train a model itself. "Build a model for it" in user language means "give me the dataset I'd need to build that model," not "train and return a trained model." Do not scope-creep into actual model training.

## Track
Wildcard Challenge — "Future of Work" theme (AI as a collaborator that plans, coordinates, and executes work — specifically: workflow automation / AI co-worker for the data-mining task).

## Core requirement
- Must use IBM Bob as the primary development tool.
- Must include AI as a core functional component (not just a wrapper).
- IBM Granite/watsonx were dropped from the AI layer — they require IBM Cloud access gated behind a credit card. Replaced with Groq API (free tier, no card required, fast inference on open-weight models). Note: this trades away a small amount of "Best Use of IBM Technology" bonus scoring, since Bob alone remains the mandatory IBM element — the rest of the AI layer is now provider-agnostic.

## Data types covered
1. **Text**
2. **Images**

## User interface concept: free-form prompt, not a fixed topic list
The user types a natural-language request describing what they need the data for (e.g. "I need training data about ocean pollution for a text classification model" — or literally anything). This is NOT restricted to a fixed dropdown. Three example prompts are offered as one-click quick-starts for demo reliability, but the text box accepts any input:
1. "I need training data about renewable energy"
2. "I need training data about ocean & marine life"
3. "I need training data about wildlife conservation"

## Intent-parsing agent (new, first step of the pipeline)
A Groq API call (open-weight Llama model, e.g. Llama 3.1/3.3) reads the user's free-form prompt and extracts the actual topic/keywords to search for. This extracted topic feeds into the same bounded pipeline below — the system still only ever queries arXiv/Openverse, regardless of how the topic was phrased. This is what makes the tool feel general-purpose to the user while staying technically bounded and reliable to build.

## Data sources
- **Text:** arXiv (via its documented API) — a preprint repository hosted by Cornell University, covering most topics (physics, environmental science, ecology, biology, etc.), with real institutional backing rather than open-editing. Query with two closely related search terms per topic (e.g. "renewable energy" + "solar power efficiency") to reliably surface genuine overlapping/near-duplicate papers for the dedup step to catch.
- **Images:** Openverse (built by Creative Commons) — an aggregator of openly-licensed images from many real institutions and sources (museums, NASA, Flickr Commons, etc.), each with clear licensing attached.

## Pipeline (agent chain)
1. **Intent-parsing agent** — extracts topic/keywords from the user's free-form prompt (see above).
2. **Collector agent** — pulls raw text (from arXiv, using two related search terms) and images (from Openverse) for the extracted topic.
3. **Cleaner agent** — strips formatting junk, boilerplate, broken/corrupt entries.
4. **Quality-filter agent** — drops low-value entries; use a Groq API call (open-weight Llama model) to judge relevance/coherence before keeping an entry.
5. **Dedup agent** —
   - Text: embed entries with `sentence-transformers` (`all-MiniLM-L6-v2`), compare via cosine similarity, flag/remove near-duplicates.
   - Images: perceptual hashing (`imagehash`) or CLIP embeddings, flag/remove near-duplicate images.
6. **Structurer agent** — formats the surviving clean data into a downloadable dataset (JSONL for text, an image folder + metadata file for images) plus a run report.

## Run report / dashboard output
- Entries collected (raw) vs. final clean count
- % duplicates removed (text and image, shown separately)
- **Explicit duplicate-pair proof (required, not optional):** for at least one real removed duplicate per run, display the actual pair side by side — for text, the two abstract/entry snippets plus their computed cosine similarity score (e.g. "0.88 similarity — entry B dropped"); for images, the two images plus their similarity score. This must be a real, live-computed example from the actual run, not a canned/staged graphic.
- Estimated compute/energy/water savings from training on the cleaned dataset vs. the raw one, computed via the methodology below
- Sample preview of surviving text + images

## Compute savings methodology (must be documented in the README, not just computed silently)
State this chain explicitly, in this order, so the number is defensible rather than arbitrary:
1. **% data reduction** = duplicates removed ÷ total entries collected (directly measured from the run — not an estimate).
2. **GPU-hours saved** = % data reduction × baseline GPU-hours for the run, using the standard assumption that training time scales roughly linearly with data volume for a fixed model and epoch count. Label this assumption explicitly as an approximation in the README.
3. **Energy saved (kWh)** = GPU-hours saved × rated power draw per GPU. Use NVIDIA's own published A100 datasheet figure — 300W (PCIe) or 400W (SXM) max thermal design power — and cite the datasheet by name.
4. **Cost saved ($)** = GPU-hours saved × a published cloud GPU hourly rate (cite the specific cloud provider's public pricing page used, pulled at build time so the figure is current).
5. **Water/carbon saved** = energy saved × a published grid carbon-intensity or data-center water-usage-effectiveness figure (cite the specific named source used, pulled at build time).
Every step above must cite its source in the README. Label the whole section: "Estimated savings, based on published NVIDIA A100 specifications and standard linear training-time scaling" — this framing pre-empts the "is this number made up?" question before a judge has to ask it.

## Tech stack
- **Frontend:** Next.js + React + Tailwind, deployed on Vercel
- **Backend/orchestration:** FastAPI (Python), LangChain/LangFlow for the agent chain, deployed on Render or Railway
- **AI/model layer:** sentence-transformers (text embeddings), imagehash or CLIP (image dedup), Groq API with an open-weight Llama model (intent-parsing + quality filtering)
- **Storage:** flat files (JSON/CSV + image folder) per run — no full database needed for this scope
- **Data sources:** arXiv API, Openverse API

## UI flow (single page)
1. User types a free-form prompt describing their need (or clicks one of the 3 example quick-start prompts)
2. Clicks "Start Mining"
3. Intent-parsing agent extracts the topic, shown briefly to the user for transparency ("Understood topic: renewable energy")
4. Live status log streams each pipeline step as it runs (collecting, cleaning, filtering, deduplicating — show real counts updating)
5. Final dashboard renders: stats, savings estimate, sample preview
6. Download button for the finished dataset

## Demo video constraint
Max 3 minutes. Script:
- 0:00–0:20 — Hook: state the problem in one line
- 0:20–0:45 — Type/select a prompt, hit start, show the extracted topic
- 0:45–1:45 — Show pipeline running live; show a real duplicate (text + image) caught and removed
- 1:45–2:20 — Final dashboard: before/after stats, savings estimate, sample output
- 2:20–2:50 — Fast mention of stack (Bob, Groq) + real-world impact in one sentence
- 2:50–3:00 — Close, team name

## Explicit scope cut (fully removed, not backlog)
The previously-considered "self-patching runtime supervisor" module (detecting wasteful AI model usage at runtime and auto-optimizing) is REMOVED from this project entirely. Do not build it under any circumstances. The data-mining pipeline is the only feature of this project.

## Team
5 people, collectively balanced across software/web dev, data/ML, and design/UX — no single dominant strength. Suggested role split (adjust to actual skills):
- Frontend (Next.js UI + live status log + dashboard)
- Backend/orchestration (FastAPI + agent chain wiring)
- AI/data layer (embeddings, dedup logic, Groq integration)
- Data-source integration (arXiv + Openverse API handling, cleaning)
- Demo/polish + video production (report formatting, savings-estimate math, recording/editing the 3-minute video)