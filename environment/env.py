"""
SQL Query Optimizer — OpenEnv Environment
==========================================
Real-world environment where an agent must optimize SQL queries for performance.
Implements the full OpenEnv spec: step() / reset() / state() API with typed
Pydantic models.
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from environment.models import Observation, Action, Reward, StepResult
from tasks.task_definitions import TASKS, Task
from graders.graders import get_grader

MAX_STEPS = 5   # Agent gets up to 5 attempts per episode


class SQLOptimizerEnv:
    """
    OpenEnv-compliant environment for SQL query optimization.

    The agent receives a slow SQL query plus schema & execution plan,
    then submits an optimized version. Reward is based on correctness,
    performance improvement, and SQL code quality.
    """

    def __init__(self, task_id: str = "easy_missing_index"):
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id: {task_id!r}. "
                f"Available tasks: {list(TASKS.keys())}"
            )
        self._task_id = task_id
        self._task: Task = TASKS[task_id]
        self._step_count: int = 0
        self._last_score: float = 0.0
        self._done: bool = False
        self._history: list[dict] = []

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset the environment to its initial state.
        Returns the initial observation.
        """
        self._step_count = 0
        self._last_score = 0.0
        self._done = False
        self._history = []
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """
        Take a step: submit an optimized SQL query.
        Returns (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        self._step_count += 1

        # Grade the action
        grader = get_grader(self._task_id)
        grade = grader.grade(
            original_query=self._task.original_query,
            optimized_query=action.optimized_query,
        )

        # Build reward object
        reward = Reward(
            total=grade.score,
            correctness=grade.correctness,
            performance=grade.performance,
            quality=grade.quality,
            explanation=grade.explanation,
        )

        self._last_score = grade.score

        # Episode ends when agent scores well OR runs out of steps
        solved = grade.score >= 0.85
        out_of_steps = self._step_count >= MAX_STEPS
        self._done = solved or out_of_steps

        # Record history
        self._history.append(
            {
                "step": self._step_count,
                "action_preview": action.optimized_query[:200],
                "score": grade.score,
                "explanation": grade.explanation,
            }
        )

        obs = self._make_observation()

        info = {
            "task_id": self._task_id,
            "difficulty": self._task.difficulty,
            "solved": solved,
            "steps_remaining": MAX_STEPS - self._step_count,
            "grade_detail": {
                "correctness": grade.correctness,
                "performance": grade.performance,
                "quality": grade.quality,
            },
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> dict:
        """
        Returns the full current state of the environment.
        Useful for checkpointing or inspection.
        """
        return {
            "task_id": self._task_id,
            "difficulty": self._task.difficulty,
            "step_count": self._step_count,
            "last_score": self._last_score,
            "done": self._done,
            "max_steps": MAX_STEPS,
            "history": self._history,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        return Observation(
            original_query=self._task.original_query,
            schema_ddl=self._task.schema,
            sample_data_description=self._task.sample_data_description,
            query_plan=self._task.query_plan,
            performance_hint=self._task.performance_hint,
            step_count=self._step_count,
            last_score=self._last_score,
        )
