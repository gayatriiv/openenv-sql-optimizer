"""
Pydantic models for the SQL Query Optimizer OpenEnv environment.
Defines typed Observation, Action, and Reward as required by OpenEnv spec.
"""

from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    """Structured observation returned by step() and reset()."""

    original_query: str = Field(
        description="The slow or unoptimized SQL query the agent must improve"
    )
    schema_ddl: str = Field(
        description="CREATE TABLE statements defining the database schema"
    )
    sample_data_description: str = Field(
        description="Description of table sizes and data distributions"
    )
    query_plan: str = Field(
        description="EXPLAIN output / execution plan of the original query"
    )
    performance_hint: str = Field(
        description="Hint about what category of optimization is needed"
    )
    step_count: int = Field(default=0, description="Steps taken in this episode")
    last_score: float = Field(
        default=0.0, description="Score from the previous action (0.0 on first step)"
    )


class Action(BaseModel):
    """Action submitted by the agent — an optimized SQL query string."""

    optimized_query: str = Field(
        description="The agent's rewritten, optimized SQL query"
    )


class Reward(BaseModel):
    """Reward breakdown returned with each step."""

    total: float = Field(ge=0.0, le=1.0, description="Total reward in [0.0, 1.0]")
    correctness: float = Field(
        ge=0.0, le=1.0, description="Does the query return correct results?"
    )
    performance: float = Field(
        ge=0.0, le=1.0, description="Estimated performance gain vs original"
    )
    quality: float = Field(
        ge=0.0, le=1.0, description="SQL code quality and idiomaticness"
    )
    explanation: str = Field(description="Human-readable reward breakdown")


class StepResult(BaseModel):
    """Full result returned by step()."""

    observation: Observation
    reward: Reward
    done: bool
    info: dict
