"""
Intent-parsing agent — multi-query expansion edition.

Calls Groq (Llama 3.3 70B) to extract a clean topic label and THREE
strategically complementary search queries from the user's free-form prompt.
Three overlapping queries guarantee semantic redundancy for the deduplicator.
"""

import os
import json
import re

from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key)
    return _client


_SYSTEM_PROMPT = """\
You are a data-mining intent parser. The user will describe what training data they need.
Your job is to extract:
1. A concise topic label (2–5 words, plain English, e.g. "ocean plastic pollution").
2. THREE strategically complementary search queries that together cover different angles
   of the same topic, ensuring semantic overlap when results are combined.

Example — for "Ocean plastic pollution and mitigation":
{
  "topic": "ocean plastic pollution",
  "queries": [
    "Ocean microplastic contamination data and studies",
    "Marine plastic waste recovery mechanisms",
    "Ocean ecosystem plastic debris detection"
  ]
}

Each query must:
- Address a DIFFERENT sub-angle (e.g. causes / effects / solutions, or data / mechanisms / policy)
- Be specific enough to return focused web and academic results
- Share enough vocabulary that results from all 3 will overlap semantically

Respond ONLY with a valid JSON object in this exact format — no markdown, no extra text:
{
  "topic": "<concise topic label>",
  "queries": ["<query 1>", "<query 2>", "<query 3>"]
}
"""


def parse_intent(prompt: str) -> tuple[str, list[str]]:
    """
    Parse the user's free-form prompt and return (topic, [query1, query2, query3]).
    Always returns exactly 3 queries. Falls back gracefully if LLM response is unparseable.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()

    try:
        raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw_clean)
        topic = str(data.get("topic", "")).strip()
        queries = [str(q).strip() for q in data.get("queries", []) if q]
        if topic and len(queries) >= 1:
            # Ensure exactly 3 queries — pad with topic variants if needed
            while len(queries) < 3:
                queries.append(f"{topic} research and analysis {len(queries) + 1}")
            return topic, queries[:3]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Fallback: derive 3 angle-varied queries from the raw prompt
    base = prompt.strip()[:80]
    fallback_topic = base
    return fallback_topic, [
        base,
        f"{base} research findings",
        f"{base} mechanisms and applications",
    ]
