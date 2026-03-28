"""
FastAPI server for the SQL Query Optimizer OpenEnv environment.
Exposes /reset, /step, /state endpoints as per OpenEnv HTTP spec.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from environment.env import SQLOptimizerEnv
from environment.models import Action

app = FastAPI(
    title="SQL Query Optimizer — OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents must optimize SQL queries. "
        "Covers 3 tasks: missing index (easy), N+1 pattern (medium), "
        "complex aggregation with window functions (hard)."
    ),
    version="1.0.0",
)

# ── In-memory session store (keyed by session_id) ─────────────────────────
_sessions: dict[str, SQLOptimizerEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = "easy_missing_index"
    session_id: str = "default"


class StepRequest(BaseModel):
    optimized_query: str
    session_id: str = "default"


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "sql-query-optimizer",
        "version": "1.0.0",
        "tasks": ["easy_missing_index", "medium_n_plus_one", "hard_complex_aggregation"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Initialize or reset the environment for a given task."""
    try:
        env = SQLOptimizerEnv(task_id=req.task_id)
        obs = env.reset()
        _sessions[req.session_id] = env
        return {"observation": obs.model_dump(), "session_id": req.session_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Submit an optimized SQL query and receive a reward."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active session '{req.session_id}'. Call /reset first.",
        )
    try:
        action = Action(optimized_query=req.optimized_query)
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = "default"):
    """Return the current state of an environment session."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active session '{session_id}'. Call /reset first.",
        )
    return env.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
