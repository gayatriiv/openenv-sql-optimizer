#!/usr/bin/env python3
"""
inference.py — Pre-submission baseline inference script.
Required filename per hackathon spec: must be named `inference.py` in repo root.

Reads credentials from environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use for inference
    HF_TOKEN      — Your Hugging Face / API key

Uses the OpenAI client for all LLM calls (pointed at API_BASE_URL).
Runs all 3 tasks and prints reproducible scores. Must complete in < 20 min.
"""

import sys
import os
import json

# ── Read required env vars ────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME")
HF_TOKEN     = os.environ.get("HF_TOKEN")

if not API_BASE_URL:
    print("ERROR: API_BASE_URL environment variable is not set.", file=sys.stderr)
    sys.exit(1)
if not MODEL_NAME:
    print("ERROR: MODEL_NAME environment variable is not set.", file=sys.stderr)
    sys.exit(1)
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from environment.env import SQLOptimizerEnv
from environment.models import Action

# ── OpenAI client pointed at API_BASE_URL ─────────────────────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

TASK_IDS = [
    "easy_missing_index",
    "medium_n_plus_one",
    "hard_complex_aggregation",
]

SYSTEM_PROMPT = """You are an expert database engineer and SQL optimization specialist.
You will receive a slow SQL query along with:
- The database schema (CREATE TABLE statements)
- A description of table sizes and data distributions
- The EXPLAIN / execution plan showing exactly why the query is slow
- A hint about what category of optimization is needed

Your task: Rewrite the query to be significantly faster.
Output ONLY the optimized SQL — no explanations, no markdown fences, no preamble.
Just the raw SQL that solves the performance problem."""


def build_user_prompt(obs: dict) -> str:
    return f"""## Database Schema
{obs['schema_ddl']}

## Data Description
{obs['sample_data_description']}

## Original (Slow) Query
{obs['original_query']}

## Execution Plan (EXPLAIN output)
{obs['query_plan']}

## Optimization Hint
{obs['performance_hint']}

Write the optimized SQL query now:"""


def strip_fences(sql: str) -> str:
    """Remove markdown code fences if the model added them."""
    if sql.startswith("```"):
        lines = sql.split("\n")
        return "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return sql


def run_inference() -> dict:
    results = {}

    print(f"\n{'=' * 60}")
    print(f"SQL Query Optimizer — Inference")
    print(f"API_BASE_URL : {API_BASE_URL}")
    print(f"MODEL_NAME   : {MODEL_NAME}")
    print(f"{'=' * 60}\n")

    for task_id in TASK_IDS:
        print(f"▶  Task: {task_id}")

        env = SQLOptimizerEnv(task_id=task_id)
        obs = env.reset()

        prompt = build_user_prompt(obs.model_dump())

        # All LLM calls use OpenAI client with API_BASE_URL + HF_TOKEN
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        optimized_sql = strip_fences(response.choices[0].message.content.strip())

        action = Action(optimized_query=optimized_sql)
        result = env.step(action)

        score = result.reward.total
        results[task_id] = {
            "score":       score,
            "correctness": result.reward.correctness,
            "performance": result.reward.performance,
            "quality":     result.reward.quality,
            "explanation": result.reward.explanation,
        }

        print(f"   Score:       {score:.4f}")
        print(f"   Correctness: {result.reward.correctness:.4f}")
        print(f"   Performance: {result.reward.performance:.4f}")
        print(f"   Quality:     {result.reward.quality:.4f}")
        print(f"   Detail:      {result.reward.explanation}")
        print()

    avg = sum(r["score"] for r in results.values()) / len(results)
    results["__average__"] = avg

    print(f"{'=' * 60}")
    print(f"Average score across {len(TASK_IDS)} tasks: {avg:.4f}")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    results = run_inference()

    # Write results to JSON for automated scoring
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")
