#!/usr/bin/env python3
"""
Baseline inference script for the SQL Query Optimizer OpenEnv environment.
Uses the OpenAI API client to run a model against all 3 tasks and report scores.

Usage:
    OPENAI_API_KEY=sk-... python scripts/baseline.py
    OPENAI_API_KEY=sk-... python scripts/baseline.py --model gpt-4o
    OPENAI_API_KEY=sk-... python scripts/baseline.py --model gpt-4o-mini --verbose
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from environment.env import SQLOptimizerEnv
from environment.models import Action

TASK_IDS = ["easy_missing_index", "medium_n_plus_one", "hard_complex_aggregation"]

SYSTEM_PROMPT = """You are an expert database engineer and SQL optimization specialist.
You will receive a slow SQL query along with:
- The database schema (CREATE TABLE statements)
- A description of data sizes
- The EXPLAIN / execution plan showing why it's slow
- A hint about what optimization is needed

Your task: Rewrite the query to be significantly faster.
Output ONLY the optimized SQL — no explanations, no markdown fences, no preamble.
Just the raw SQL that replaces the original query."""


def build_user_prompt(obs: dict) -> str:
    return f"""## Database Schema
{obs['schema_ddl']}

## Sample Data Description
{obs['sample_data_description']}

## Original (Slow) Query
{obs['original_query']}

## Execution Plan (EXPLAIN output)
{obs['query_plan']}

## Optimization Hint
{obs['performance_hint']}

Write the optimized SQL query now:"""


def run_baseline(model: str = "gpt-4o-mini", verbose: bool = False) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results = {}

    print(f"\n{'='*60}")
    print(f"SQL Query Optimizer — Baseline Inference")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    for task_id in TASK_IDS:
        print(f"▶  Task: {task_id}")
        env = SQLOptimizerEnv(task_id=task_id)
        obs = env.reset()

        user_prompt = build_user_prompt(obs.model_dump())

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )

        optimized_sql = response.choices[0].message.content.strip()
        # Strip markdown fences if model added them
        if optimized_sql.startswith("```"):
            lines = optimized_sql.split("\n")
            optimized_sql = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            ).strip()

        action = Action(optimized_query=optimized_sql)
        result = env.step(action)

        score = result.reward.total
        results[task_id] = {
            "score": score,
            "correctness": result.reward.correctness,
            "performance": result.reward.performance,
            "quality": result.reward.quality,
            "explanation": result.reward.explanation,
        }

        print(f"   Score:       {score:.4f}")
        print(f"   Correctness: {result.reward.correctness:.4f}")
        print(f"   Performance: {result.reward.performance:.4f}")
        print(f"   Quality:     {result.reward.quality:.4f}")
        print(f"   Explanation: {result.reward.explanation}")
        if verbose:
            print(f"\n   --- Optimized SQL ---\n{optimized_sql}\n")
        print()

    avg = sum(r["score"] for r in results.values()) / len(results)
    print(f"{'='*60}")
    print(f"Average score across 3 tasks: {avg:.4f}")
    print(f"{'='*60}\n")

    results["__average__"] = avg
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Optimizer baseline inference")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--verbose", action="store_true", help="Print generated SQL")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    results = run_baseline(model=args.model, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
