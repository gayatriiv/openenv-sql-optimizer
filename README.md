# openenv-sql-optimizer

> **A real-world OpenEnv environment for training and evaluating AI agents on SQL query optimization.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-4ade80?style=flat-square)](https://openenv.dev)
[![Python](https://img.shields.io/badge/python-3.12+-3b82f6?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-20%20passed-22c55e?style=flat-square)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-f59e0b?style=flat-square)](LICENSE)

---

SQL optimization is something every production engineering team does — and gets wrong. This environment puts an agent in the shoes of a database engineer: given a slow query, its schema, and an execution plan, **rewrite it to be dramatically faster**.

Three tasks. Real anti-patterns. Partial credit rewards. No games, no toys.

```
Task 1 (Easy)   →  2M-row full table scan from a missing email index
Task 2 (Medium) →  120,000 database round-trips from an N+1 query pattern  
Task 3 (Hard)   →  45-second BI report with correlated subqueries + non-SARGable predicates
```

---

## Table of Contents

- [Why This Environment?](#why-this-environment)
- [Quickstart](#quickstart)
- [Tasks](#tasks)
- [API Reference](#api-reference)
- [Reward Function](#reward-function)
- [Baseline Scores](#baseline-scores)
- [Project Structure](#project-structure)
- [Docker & Deployment](#docker--deployment)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Why This Environment?

Most agent benchmarks use synthetic puzzles. This one doesn't.

SQL optimization is a **genuine, high-value, measurable task** with a natural difficulty gradient and clear success criteria. Agents that score well here have learned something transferable — they understand indexes, join strategies, and query plan analysis, not just pattern matching on toy inputs.

| Property | This environment |
|---|---|
| Real-world task | ✅ Database engineers do this daily |
| Measurable success | ✅ Did it add the index? Collapse the N+1? Use `LAG()`? |
| Partial credit | ✅ Reward on every step, not just at episode end |
| Difficulty range | ✅ Easy → Medium → Hard with genuine progression |
| Deterministic grading | ✅ Same input always produces the same score |

---

## Quickstart

### Python (direct)

```bash
git clone https://github.com/your-username/openenv-sql-optimizer
cd openenv-sql-optimizer
pip install -r requirements.txt
```

```python
from environment.env import SQLOptimizerEnv
from environment.models import Action

env = SQLOptimizerEnv("easy_missing_index")
obs = env.reset()

print(obs.original_query)
# → SELECT u.id, u.username ... WHERE u.email = 'alice@example.com'

print(obs.performance_hint)
# → Adding a unique index on email would turn the full scan into a single-row lookup.

action = Action(optimized_query="""
    CREATE UNIQUE INDEX idx_users_email ON users (email);

    SELECT u.id, u.username, u.email, p.name AS plan_name
    FROM   users u
    LEFT   JOIN plans p ON p.id = u.plan_id
    WHERE  u.email = 'alice@example.com'
      AND  u.status = 'active';
""")

result = env.step(action)
print(f"Score: {result.reward.total:.2f}")        # → Score: 1.00
print(f"Done:  {result.done}")                     # → Done: True
print(result.reward.explanation)
# → ✓ Keeps email filter | ✓ Keeps status filter | ✓ Uses UNIQUE index (optimal, +1.0)
```

### HTTP API

```bash
python server.py  # starts on http://localhost:7860
```

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium_n_plus_one", "session_id": "agent-1"}'

# Submit an optimized query
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "optimized_query": "SELECT o.id, o.amount_cents, c.name, c.email FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.status = '\''completed'\'' AND o.created_at >= '\''2024-01-01'\''",
    "session_id": "agent-1"
  }'

# Inspect state
curl "http://localhost:7860/state?session_id=agent-1"
```

### Docker

```bash
docker build -t openenv-sql-optimizer .
docker run -p 7860:7860 openenv-sql-optimizer
```

---

## Tasks

Each task presents the agent with a slow query, the full database schema, a simulated `EXPLAIN` plan, and a hint about the category of fix needed. The agent has up to **5 attempts** per episode.

---

### Task 1 — `easy_missing_index`
**Difficulty: Easy | Expected score: 0.7–1.0**

A customer authentication query filters by `email` on a 2-million-row `users` table. No index exists on that column, so every call triggers a full table scan through 2M rows to return one.

```sql
-- ❌ Original: full table scan (201,000+ row read cost)
SELECT u.id, u.username, u.email, p.name AS plan_name
FROM   users u
LEFT   JOIN plans p ON p.id = u.plan_id
WHERE  u.email = 'alice@example.com'
  AND  u.status = 'active';
```

```
EXPLAIN output:
→ Table scan on u  (cost=201003 rows=2010033)  ← FULL SCAN
```

**What a perfect solution looks like:** Add `CREATE UNIQUE INDEX` on `users(email)` and preserve the original query logic. Turns a 200K-row scan into a single-row primary key lookup.

---

### Task 2 — `medium_n_plus_one`
**Difficulty: Medium | Expected score: 0.5–0.9**

Application code fetches all completed orders, then fires one `SELECT` per order to look up the customer name — the classic N+1 problem. With 120,000 matching orders, this is 120,001 individual round trips to the database.

```sql
-- ❌ Original: two queries, second fires 120,000 times
-- Query 1: fetch all completed orders
SELECT id, customer_id, amount_cents, status FROM orders
WHERE  status = 'completed' AND created_at >= '2024-01-01';

-- Query 2: fired once per row from Query 1
SELECT name, email FROM customers WHERE id = ?;
```

**What a perfect solution looks like:** A single `JOIN` between `orders` and `customers` that collapses 120,001 round trips into one.

---

### Task 3 — `hard_complex_aggregation`
**Difficulty: Hard | Expected score: 0.3–0.8**

A business intelligence query computes monthly revenue by product category, including a month-over-month comparison. It runs for **45+ seconds** due to two compounding problems:

1. **Non-SARGable predicates** — `YEAR(order_date)` and `MONTH(order_date)` prevent the query engine from using the index on `order_date`, forcing a full scan of 15M rows.
2. **Correlated subquery** — The previous-month comparison reruns a 15M-row scan once per (month, category) pair — 144 times total.

```sql
-- ❌ Original: 45+ seconds
SELECT
    YEAR(s.order_date) AS year, MONTH(s.order_date) AS month,
    p.category,
    SUM(s.quantity * s.unit_price * (1 - s.discount_pct/100)) AS revenue,
    (
        -- correlated subquery: full scan × 144
        SELECT SUM(s2.quantity * s2.unit_price * (1 - s2.discount_pct/100))
        FROM   sales s2 JOIN products p2 ON p2.id = s2.product_id
        WHERE  p2.category = p.category
          AND  YEAR(s2.order_date)  = YEAR(s.order_date)
          AND  MONTH(s2.order_date) = MONTH(s.order_date) - 1
    ) AS prev_month_revenue
FROM   sales s JOIN products p ON p.id = s.product_id
WHERE  YEAR(s.order_date) = 2024
GROUP  BY YEAR(s.order_date), MONTH(s.order_date), p.category;
```

**What a perfect solution looks like:** A CTE that aggregates revenue once with a SARGable date range, then a `LAG()` window function over the CTE result to get the previous month — total: one pass over the data.

```sql
-- ✅ Optimized: ~1.2 seconds
WITH monthly AS (
    SELECT DATE_FORMAT(s.order_date, '%Y-%m-01') AS month_start,
           p.category,
           SUM(s.quantity * s.unit_price * (1 - s.discount_pct/100)) AS revenue
    FROM   sales s JOIN products p ON p.id = s.product_id
    WHERE  s.order_date >= '2024-01-01' AND s.order_date < '2025-01-01'
    GROUP  BY month_start, p.category
)
SELECT month_start, category, revenue,
       LAG(revenue) OVER (PARTITION BY category ORDER BY month_start) AS prev_month_revenue
FROM   monthly
ORDER  BY month_start, category;
```

---

## API Reference

### OpenEnv Models

```python
class Observation(BaseModel):
    original_query: str           # The slow SQL to optimize
    schema_ddl: str               # CREATE TABLE statements
    sample_data_description: str  # Table sizes and data distributions
    query_plan: str               # EXPLAIN output of the original query
    performance_hint: str         # Category of optimization needed
    step_count: int               # Steps taken this episode (max 5)
    last_score: float             # Score from previous step (0.0 on first)

class Action(BaseModel):
    optimized_query: str          # Agent's rewritten SQL (may include DDL)

class Reward(BaseModel):
    total: float        # Weighted total [0.0, 1.0]
    correctness: float  # Query still returns correct results?
    performance: float  # Addresses the actual bottleneck?
    quality: float      # Idiomatic, clean SQL?
    explanation: str    # Human-readable breakdown of each component
```

### Environment Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `reset()` | `Observation` | Initializes the episode, returns the first observation |
| `step(action)` | `StepResult` | Submits a query, returns observation + reward + done + info |
| `state()` | `dict` | Returns full current state for checkpointing or inspection |

### HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start or restart an episode |
| `POST` | `/step` | Submit an optimized query |
| `GET` | `/state` | Inspect current session state |
| `GET` | `/health` | Health check |
| `GET` | `/` | Environment metadata |

---

## Reward Function

Reward is computed on **every step**, not just at episode end. This gives agents a useful learning signal throughout the trajectory.

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Correctness** | 35% | Does the query preserve the original filters, joins, and columns? |
| **Performance** | 50% | Does it address the specific bottleneck? (index added / N+1 collapsed / window function used) |
| **Quality** | 15% | Is the SQL idiomatic? (penalizes `SELECT *`, rewards clean structure) |

```
total = 0.35 × correctness + 0.50 × performance + 0.15 × quality
```

Graders are **deterministic** — the same query always produces the same score. An episode is considered solved when `total ≥ 0.85`. Agents get up to 5 attempts per episode.

---

## Baseline Scores

Run with `gpt-4o-mini` and `gpt-4o` at temperature 0 using the included baseline script:

| Task | Difficulty | gpt-4o-mini | gpt-4o |
|------|------------|:-----------:|:------:|
| `easy_missing_index` | Easy | 0.82 | 0.91 |
| `medium_n_plus_one` | Medium | 0.61 | 0.78 |
| `hard_complex_aggregation` | Hard | 0.44 | 0.63 |
| **Average** | | **0.62** | **0.77** |

### Run the baseline yourself

```bash
OPENAI_API_KEY=sk-... python scripts/baseline.py --model gpt-4o-mini
OPENAI_API_KEY=sk-... python scripts/baseline.py --model gpt-4o --verbose
OPENAI_API_KEY=sk-... python scripts/baseline.py --model gpt-4o --output results.json
```

---

## Project Structure

```
openenv-sql-optimizer/
│
├── openenv.yaml              # OpenEnv metadata (name, tasks, action/observation spaces)
├── server.py                 # FastAPI HTTP server (port 7860)
├── requirements.txt
├── Dockerfile
├── README.md
│
├── environment/
│   ├── models.py             # Pydantic models: Observation, Action, Reward, StepResult
│   └── env.py                # SQLOptimizerEnv — step() / reset() / state()
│
├── tasks/
│   └── task_definitions.py   # All 3 task scenarios (schema, query, plan, hints)
│
├── graders/
│   └── graders.py            # Deterministic graders for easy / medium / hard
│
├── scripts/
│   └── baseline.py           # OpenAI API baseline inference + scoring
│
└── tests/
    └── test_env.py           # 20-test suite covering models, API, and all graders
```

---

## Docker & Deployment

### Local Docker

```bash
docker build -t openenv-sql-optimizer .
docker run -p 7860:7860 openenv-sql-optimizer

# Verify it's running
curl http://localhost:7860/health
# → {"status": "ok"}
```

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Docker**
3. Push this repository to the Space
4. Tag the Space with `openenv`

The server binds to `0.0.0.0:7860` by default, which is what HF Spaces expects.

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

```
tests/test_env.py::TestModels::test_action_validation          PASSED
tests/test_env.py::TestModels::test_observation_fields         PASSED
tests/test_env.py::TestModels::test_reward_range               PASSED
tests/test_env.py::TestEnvironmentAPI::test_all_tasks_loadable PASSED
tests/test_env.py::TestEnvironmentAPI::test_episode_ends_...   PASSED
tests/test_env.py::TestEnvironmentAPI::test_invalid_task_...   PASSED
tests/test_env.py::TestEnvironmentAPI::test_reset_after_done   PASSED
tests/test_env.py::TestEnvironmentAPI::test_reset_returns_...  PASSED
tests/test_env.py::TestEnvironmentAPI::test_state_returns_dict PASSED
tests/test_env.py::TestEnvironmentAPI::test_step_returns_...   PASSED
tests/test_env.py::TestGraders::test_easy_bad_syntax           PASSED
tests/test_env.py::TestGraders::test_easy_no_index             PASSED
tests/test_env.py::TestGraders::test_easy_perfect_solution     PASSED
tests/test_env.py::TestGraders::test_easy_score_range          PASSED
tests/test_env.py::TestGraders::test_graders_deterministic     PASSED
tests/test_env.py::TestGraders::test_hard_original_unchanged   PASSED
tests/test_env.py::TestGraders::test_hard_perfect_solution     PASSED
tests/test_env.py::TestGraders::test_hard_score_range          PASSED
tests/test_env.py::TestGraders::test_medium_perfect_solution   PASSED
tests/test_env.py::TestGraders::test_medium_still_n_plus_one   PASSED

20 passed in 0.27s
```

---

## Contributing

Contributions are welcome — especially new tasks. A good task has:

- A realistic scenario (something a real engineer would encounter)
- Clear bottleneck that can be detected programmatically
- Partial-credit grading (not just binary pass/fail)
- A hard case that genuinely challenges frontier models

Open an issue to discuss a new task before implementing it.

---

## License

MIT — see [LICENSE](LICENSE).
