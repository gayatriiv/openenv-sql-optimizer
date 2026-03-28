---
title: SQL Query Optimizer — OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - sql
  - reinforcement-learning
  - agent-evaluation
  - databases
license: mit
---

# SQL Query Optimizer — OpenEnv Environment

A **real-world OpenEnv environment** where AI agents must optimize SQL queries for performance. This environment simulates the everyday work of a database engineer: analyzing slow queries, identifying inefficiencies, and rewriting them to be dramatically faster.

## Why SQL Optimization?

SQL optimization is a genuine, high-value task that:
- Every production backend team faces regularly
- Has clear, measurable success criteria (does it hit the index? remove the N+1?)
- Spans a wide difficulty range (trivial missing index → complex window function rewrites)
- Is immediately useful for evaluating and training code-generation agents

---

## Environment Description

The agent receives a **slow SQL query** plus its database schema and `EXPLAIN` execution plan, then must submit an optimized version. Reward is based on three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 35% | Does the query still return the right results? (preserves filters, JOINs, columns) |
| Performance | 50% | Does it address the actual bottleneck? (index added, N+1 eliminated, window fn used) |
| Quality | 15% | Is the SQL idiomatic and clean? (no `SELECT *`, reasonable length) |

---

## Action & Observation Spaces

### Observation (returned by `reset()` and `step()`)

```python
class Observation(BaseModel):
    original_query: str          # The slow SQL query to optimize
    schema_ddl: str              # CREATE TABLE statements
    sample_data_description: str # Table sizes and data distribution info
    query_plan: str              # EXPLAIN output showing why it's slow
    performance_hint: str        # Category of optimization needed
    step_count: int              # Steps taken this episode (max 5)
    last_score: float            # Score from previous action (0.0 on first step)
```

### Action (submitted via `step()`)

```python
class Action(BaseModel):
    optimized_query: str  # The agent's rewritten SQL (may include DDL like CREATE INDEX)
```

### Reward (returned with each step)

```python
class Reward(BaseModel):
    total: float        # Weighted total score [0.0, 1.0]
    correctness: float  # Correctness component [0.0, 1.0]
    performance: float  # Performance component [0.0, 1.0]
    quality: float      # Quality component [0.0, 1.0]
    explanation: str    # Human-readable breakdown
```

---

## Tasks

### Task 1 — `easy_missing_index` (Easy)
**Missing index causing full table scan**

A customer lookup query filters by `email` on a 2M-row `users` table, but there's no index on that column. The full table scan costs 200K+ row reads when it should be a single-row lookup.

- **Goal:** Add `CREATE [UNIQUE] INDEX` on `users.email` AND preserve the query logic
- **Expected score range:** 0.7–1.0
- **Baseline (gpt-4o-mini):** 0.82

### Task 2 — `medium_n_plus_one` (Medium)
**N+1 query pattern — 120,000 database round trips**

Application code fetches all completed orders, then fires a separate `SELECT` per order to get the customer name — the classic N+1 problem. With 120,000 matching orders, this generates 120,001 database queries.

- **Goal:** Collapse into a single `JOIN` query
- **Expected score range:** 0.5–0.9
- **Baseline (gpt-4o-mini):** 0.61

### Task 3 — `hard_complex_aggregation` (Hard)
**45-second BI report with correlated subqueries + non-SARGable predicates**

A business intelligence query computes monthly revenue by category using:
1. **Non-SARGable predicates** — `YEAR(order_date)` and `MONTH(order_date)` prevent index seeks
2. **Correlated subquery** for previous month comparison — runs 144 full table scans (12 months × 12 categories × 15M rows each)

- **Goal:** Rewrite using a CTE + `LAG()` window function + date range predicates
- **Expected score range:** 0.3–0.8
- **Baseline (gpt-4o-mini):** 0.44

---

## Baseline Scores

| Task | Difficulty | gpt-4o-mini | gpt-4o |
|------|------------|-------------|--------|
| easy_missing_index | Easy | 0.82 | 0.91 |
| medium_n_plus_one | Medium | 0.61 | 0.78 |
| hard_complex_aggregation | Hard | 0.44 | 0.63 |
| **Average** | | **0.62** | **0.77** |

---

## Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/your-username/sql-query-optimizer
cd sql-query-optimizer
pip install -r requirements.txt
python server.py
```

### Docker

```bash
docker build -t sql-optimizer-env .
docker run -p 7860:7860 sql-optimizer-env
```

The server starts at `http://localhost:7860`.

### HTTP API

```bash
# Reset to a task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_missing_index", "session_id": "my-agent"}'

# Submit an optimized query
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "optimized_query": "CREATE UNIQUE INDEX idx_users_email ON users(email);\nSELECT u.id, u.username, u.email, p.name FROM users u LEFT JOIN plans p ON p.id = u.plan_id WHERE u.email = '"'"'alice@example.com'"'"' AND u.status = '"'"'active'"'"';",
    "session_id": "my-agent"
  }'

# Inspect state
curl http://localhost:7860/state?session_id=my-agent
```

### Python SDK (direct)

```python
from environment.env import SQLOptimizerEnv
from environment.models import Action

env = SQLOptimizerEnv("hard_complex_aggregation")
obs = env.reset()

print(obs.original_query)
print(obs.performance_hint)

action = Action(optimized_query="""
WITH monthly AS (
    SELECT DATE_FORMAT(s.order_date, '%Y-%m-01') AS month_start,
           p.category,
           SUM(s.quantity * s.unit_price * (1 - s.discount_pct/100)) AS revenue
    FROM   sales s
    JOIN   products p ON p.id = s.product_id
    WHERE  s.order_date >= '2024-01-01' AND s.order_date < '2025-01-01'
    GROUP  BY month_start, p.category
)
SELECT month_start, category, revenue,
       LAG(revenue) OVER (PARTITION BY category ORDER BY month_start) AS prev_month_revenue
FROM   monthly
ORDER  BY month_start, category;
""")

result = env.step(action)
print(f"Score: {result.reward.total:.3f}")
print(f"Breakdown: {result.reward.explanation}")
```

### Baseline Script

```bash
OPENAI_API_KEY=sk-... python scripts/baseline.py --model gpt-4o --verbose
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
sql-query-optimizer/
├── openenv.yaml              # OpenEnv metadata
├── server.py                 # FastAPI HTTP server
├── requirements.txt
├── Dockerfile
├── README.md
├── environment/
│   ├── __init__.py
│   ├── models.py             # Pydantic: Observation, Action, Reward, StepResult
│   └── env.py                # SQLOptimizerEnv — step() / reset() / state()
├── tasks/
│   ├── __init__.py
│   └── task_definitions.py   # Easy, Medium, Hard task scenarios
├── graders/
│   ├── __init__.py
│   └── graders.py            # Deterministic graders for all 3 tasks
├── scripts/
│   └── baseline.py           # OpenAI API baseline inference script
└── tests/
    └── test_env.py           # 20-test suite (all passing)
```

---

## License

MIT
