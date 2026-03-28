#!/usr/bin/env python3
"""
Test suite for the SQL Query Optimizer OpenEnv environment.
Tests step(), reset(), state(), graders, and model validation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from environment.env import SQLOptimizerEnv
from environment.models import Action, Observation, Reward
from graders.graders import (
    EasyMissingIndexGrader,
    MediumNPlusOneGrader,
    HardComplexAggregationGrader,
    get_grader,
)


class TestModels(unittest.TestCase):
    """Test Pydantic model validation."""

    def test_observation_fields(self):
        obs = Observation(
            original_query="SELECT 1",
            schema_ddl="CREATE TABLE t (id INT)",
            sample_data_description="100 rows",
            query_plan="-> Full scan",
            performance_hint="Add index",
        )
        self.assertEqual(obs.step_count, 0)
        self.assertEqual(obs.last_score, 0.0)

    def test_action_validation(self):
        a = Action(optimized_query="SELECT id FROM users WHERE email='x@x.com'")
        self.assertIn("SELECT", a.optimized_query)

    def test_reward_range(self):
        r = Reward(
            total=0.75,
            correctness=0.9,
            performance=0.6,
            quality=1.0,
            explanation="test",
        )
        self.assertGreaterEqual(r.total, 0.0)
        self.assertLessEqual(r.total, 1.0)


class TestEnvironmentAPI(unittest.TestCase):
    """Test the core step/reset/state API."""

    def test_reset_returns_observation(self):
        env = SQLOptimizerEnv("easy_missing_index")
        obs = env.reset()
        self.assertIsInstance(obs, Observation)
        self.assertIn("users", obs.schema_ddl)
        self.assertEqual(obs.step_count, 0)
        self.assertEqual(obs.last_score, 0.0)

    def test_step_returns_step_result(self):
        env = SQLOptimizerEnv("easy_missing_index")
        env.reset()
        action = Action(optimized_query="SELECT id FROM users WHERE email='x@x.com'")
        result = env.step(action)
        self.assertIsInstance(result.reward, Reward)
        self.assertGreaterEqual(result.reward.total, 0.0)
        self.assertLessEqual(result.reward.total, 1.0)
        self.assertFalse(result.done)  # score too low to be done

    def test_state_returns_dict(self):
        env = SQLOptimizerEnv("medium_n_plus_one")
        env.reset()
        s = env.state()
        self.assertEqual(s["task_id"], "medium_n_plus_one")
        self.assertEqual(s["step_count"], 0)
        self.assertFalse(s["done"])

    def test_episode_ends_after_max_steps(self):
        env = SQLOptimizerEnv("easy_missing_index")
        env.reset()
        action = Action(optimized_query="SELECT 1")
        for _ in range(5):
            result = env.step(action)
        self.assertTrue(result.done)

    def test_reset_after_done(self):
        env = SQLOptimizerEnv("easy_missing_index")
        env.reset()
        action = Action(optimized_query="SELECT 1")
        for _ in range(5):
            env.step(action)
        # Should raise if we step again without reset
        with self.assertRaises(RuntimeError):
            env.step(action)
        # But reset should work
        obs = env.reset()
        self.assertEqual(obs.step_count, 0)

    def test_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            SQLOptimizerEnv("nonexistent_task")

    def test_all_tasks_loadable(self):
        for task_id in ["easy_missing_index", "medium_n_plus_one", "hard_complex_aggregation"]:
            env = SQLOptimizerEnv(task_id)
            obs = env.reset()
            self.assertIsNotNone(obs.original_query)


class TestGraders(unittest.TestCase):
    """Test grader logic and score ranges."""

    def _grade(self, task_id, sql):
        g = get_grader(task_id)
        from tasks.task_definitions import TASKS
        return g.grade(TASKS[task_id].original_query, sql)

    # ── Easy grader ──────────────────────────────────────────────────────

    def test_easy_perfect_solution(self):
        sql = """
        CREATE UNIQUE INDEX idx_users_email ON users (email);
        SELECT u.id, u.username, u.email, p.name AS plan_name
        FROM   users u
        LEFT   JOIN plans p ON p.id = u.plan_id
        WHERE  u.email = 'alice@example.com'
          AND  u.status = 'active';
        """
        result = self._grade("easy_missing_index", sql)
        self.assertGreater(result.score, 0.8)

    def test_easy_no_index(self):
        sql = "SELECT * FROM users WHERE email = 'alice@example.com'"
        result = self._grade("easy_missing_index", sql)
        self.assertLess(result.score, 0.5)

    def test_easy_bad_syntax(self):
        result = self._grade("easy_missing_index", "NOT VALID SQL !!!")
        self.assertEqual(result.score, 0.0)

    def test_easy_score_range(self):
        for sql in [
            "SELECT id FROM users WHERE email='x'",
            "CREATE INDEX i ON users(email); SELECT id FROM users WHERE email='x' AND status='active'",
        ]:
            r = self._grade("easy_missing_index", sql)
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0)

    # ── Medium grader ─────────────────────────────────────────────────────

    def test_medium_perfect_solution(self):
        sql = """
        SELECT o.id, o.amount_cents, o.status, o.created_at,
               c.name, c.email
        FROM   orders o
        JOIN   customers c ON c.id = o.customer_id
        WHERE  o.status = 'completed'
          AND  o.created_at >= '2024-01-01';
        """
        result = self._grade("medium_n_plus_one", sql)
        self.assertGreater(result.score, 0.7)

    def test_medium_still_n_plus_one(self):
        sql = "SELECT id FROM orders WHERE status='completed'; SELECT name FROM customers WHERE id = ?"
        result = self._grade("medium_n_plus_one", sql)
        self.assertLess(result.score, 0.5)

    # ── Hard grader ───────────────────────────────────────────────────────

    def test_hard_perfect_solution(self):
        sql = """
        WITH monthly_revenue AS (
            SELECT
                DATE_FORMAT(s.order_date, '%Y-%m-01') AS month_start,
                p.category,
                SUM(s.quantity * s.unit_price * (1 - s.discount_pct / 100)) AS revenue
            FROM   sales s
            JOIN   products p ON p.id = s.product_id
            WHERE  s.order_date >= '2024-01-01' AND s.order_date < '2025-01-01'
            GROUP  BY month_start, p.category
        )
        SELECT
            month_start,
            category,
            revenue,
            LAG(revenue) OVER (PARTITION BY category ORDER BY month_start) AS prev_month_revenue
        FROM monthly_revenue
        ORDER BY month_start, category;
        """
        result = self._grade("hard_complex_aggregation", sql)
        self.assertGreater(result.score, 0.7)

    def test_hard_original_unchanged(self):
        from tasks.task_definitions import TASKS
        original = TASKS["hard_complex_aggregation"].original_query
        result = self._grade("hard_complex_aggregation", original)
        # Original uses non-SARGable YEAR() and correlated subquery — should score low
        self.assertLessEqual(result.score, 0.5)

    def test_hard_score_range(self):
        sql = "SELECT category, SUM(quantity * unit_price) FROM sales JOIN products p ON p.id = product_id WHERE order_date >= '2024-01-01' GROUP BY category"
        r = self._grade("hard_complex_aggregation", sql)
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 1.0)

    # ── Determinism ───────────────────────────────────────────────────────

    def test_graders_deterministic(self):
        """Same input always produces same output."""
        sql = "CREATE UNIQUE INDEX idx ON users(email); SELECT id FROM users WHERE email='x' AND status='active'"
        r1 = self._grade("easy_missing_index", sql)
        r2 = self._grade("easy_missing_index", sql)
        self.assertEqual(r1.score, r2.score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
