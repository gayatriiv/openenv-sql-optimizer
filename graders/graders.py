"""
Programmatic graders for the SQL Query Optimizer environment.
Each grader scores an agent's optimized query on a 0.0–1.0 scale.
Graders are deterministic and reproducible.
"""

import re
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GradeResult:
    score: float          # 0.0 – 1.0
    correctness: float    # 0.0 – 1.0
    performance: float    # 0.0 – 1.0
    quality: float        # 0.0 – 1.0
    explanation: str


class BaseGrader(ABC):
    """Abstract base grader. All graders must implement grade()."""

    @abstractmethod
    def grade(self, original_query: str, optimized_query: str) -> GradeResult:
        """Score the optimized query. Returns GradeResult with score in [0.0, 1.0]."""
        ...

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _normalize(self, sql: str) -> str:
        """Lowercase + collapse whitespace for pattern matching."""
        return re.sub(r"\s+", " ", sql.lower().strip())

    def _has_pattern(self, sql: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
        return bool(re.search(pattern, sql, flags))

    def _syntax_ok(self, sql: str) -> bool:
        """Light syntax check — just look for basic SQL structure."""
        norm = self._normalize(sql)
        # Must have at least SELECT ... FROM
        return bool(re.search(r"\bselect\b.+\bfrom\b", norm, re.DOTALL))

    def _count_table_scans(self, query: str) -> int:
        """Heuristic: count non-SARGable function wraps on indexed columns."""
        norm = self._normalize(query)
        scans = 0
        scans += len(re.findall(r"\byear\s*\(", norm))
        scans += len(re.findall(r"\bmonth\s*\(", norm))
        scans += len(re.findall(r"\bdate\s*\(", norm))
        return scans

    def _quality_score(self, query: str) -> float:
        """Heuristic SQL quality checks."""
        norm = self._normalize(query)
        score = 1.0
        if re.search(r"\bselect\s+\*", norm):
            score -= 0.15   # avoid SELECT *
        if not re.search(r"\bwhere\b", norm) and "join" not in norm:
            score -= 0.1    # no filtering at all
        # Reasonable length
        if len(query) < 20:
            score -= 0.3
        return max(0.0, min(1.0, score))


# ── Easy: Missing Index ────────────────────────────────────────────────────

class EasyMissingIndexGrader(BaseGrader):
    """
    Grades solutions to the missing-index task.
    A perfect solution: adds CREATE [UNIQUE] INDEX on users.email AND keeps SELECT correct.
    """

    def grade(self, original_query: str, optimized_query: str) -> GradeResult:
        parts = []
        explanation_parts = []

        # 1. Syntax check
        if not self._syntax_ok(optimized_query):
            return GradeResult(
                score=0.0, correctness=0.0, performance=0.0, quality=0.0,
                explanation="Syntax error: could not find SELECT ... FROM structure."
            )

        # 2. Correctness: still selects the right columns
        norm = self._normalize(optimized_query)
        has_email_filter = self._has_pattern(norm, r"u\.email\s*=|email\s*=")
        has_status_filter = self._has_pattern(norm, r"status\s*=")
        has_join_or_left = self._has_pattern(norm, r"\bjoin\b")
        correctness = 0.0
        if has_email_filter:
            correctness += 0.5
            explanation_parts.append("✓ Keeps email filter")
        else:
            explanation_parts.append("✗ Missing email filter (loses correctness)")
        if has_status_filter:
            correctness += 0.3
            explanation_parts.append("✓ Keeps status filter")
        if has_join_or_left:
            correctness += 0.2
            explanation_parts.append("✓ Keeps JOIN to plans")

        # 3. Performance: does it add an index on email?
        has_index = self._has_pattern(
            optimized_query,
            r"create\s+(unique\s+)?index.+on\s+users\s*\(\s*email",
        )
        has_unique_index = self._has_pattern(
            optimized_query,
            r"create\s+unique\s+index.+on\s+users\s*\(\s*email",
        )
        performance = 0.0
        if has_index:
            performance = 0.8
            explanation_parts.append("✓ Adds index on users.email (+0.8)")
        if has_unique_index:
            performance = 1.0
            explanation_parts.append("✓ Uses UNIQUE index (optimal, +1.0)")
        if not has_index:
            explanation_parts.append("✗ No index created on users.email")

        # 4. Quality
        quality = self._quality_score(optimized_query)
        if quality < 1.0:
            explanation_parts.append(f"Quality deduction: {1-quality:.2f}")

        # Weighted total: correctness 35%, performance 50%, quality 15%
        total = 0.35 * correctness + 0.50 * performance + 0.15 * quality
        total = round(min(1.0, max(0.0, total)), 4)

        return GradeResult(
            score=total,
            correctness=round(correctness, 4),
            performance=round(performance, 4),
            quality=round(quality, 4),
            explanation=" | ".join(explanation_parts),
        )


# ── Medium: N+1 Problem ────────────────────────────────────────────────────

class MediumNPlusOneGrader(BaseGrader):
    """
    Grades solutions to the N+1 query task.
    Perfect solution: single query with JOIN between orders and customers.
    """

    def grade(self, original_query: str, optimized_query: str) -> GradeResult:
        explanation_parts = []

        if not self._syntax_ok(optimized_query):
            return GradeResult(
                score=0.0, correctness=0.0, performance=0.0, quality=0.0,
                explanation="Syntax error: could not find SELECT ... FROM structure."
            )

        norm = self._normalize(optimized_query)

        # Correctness
        has_orders = self._has_pattern(norm, r"\borders\b")
        has_customers = self._has_pattern(norm, r"\bcustomers\b")
        has_status_filter = self._has_pattern(norm, r"status\s*=\s*['\"]completed['\"]")
        has_date_filter = self._has_pattern(norm, r"created_at\s*(>=|>|between)")

        correctness = 0.0
        if has_orders:
            correctness += 0.3
            explanation_parts.append("✓ References orders table")
        if has_customers:
            correctness += 0.3
            explanation_parts.append("✓ References customers table")
        if has_status_filter:
            correctness += 0.2
            explanation_parts.append("✓ Keeps status='completed' filter")
        if has_date_filter:
            correctness += 0.2
            explanation_parts.append("✓ Keeps date range filter")

        # Performance: single JOIN vs multiple queries
        has_join = self._has_pattern(norm, r"\bjoin\b.+customers|customers.+\bjoin\b")
        is_single_statement = optimized_query.strip().count(";") <= 1
        still_parameterized = self._has_pattern(norm, r"where\s+id\s*=\s*\?")

        performance = 0.0
        if has_join:
            performance += 0.7
            explanation_parts.append("✓ Uses JOIN (eliminates N+1)")
        if is_single_statement:
            performance += 0.2
            explanation_parts.append("✓ Single query statement")
        if still_parameterized:
            performance = max(0.0, performance - 0.5)
            explanation_parts.append("✗ Still uses parameterized per-row lookup")
        # Bonus: also joins products
        if self._has_pattern(norm, r"\bproducts\b"):
            performance = min(1.0, performance + 0.1)
            explanation_parts.append("✓ Bonus: also joins products table")

        quality = self._quality_score(optimized_query)

        total = 0.35 * correctness + 0.50 * performance + 0.15 * quality
        total = round(min(1.0, max(0.0, total)), 4)

        return GradeResult(
            score=total,
            correctness=round(correctness, 4),
            performance=round(performance, 4),
            quality=round(quality, 4),
            explanation=" | ".join(explanation_parts),
        )


# ── Hard: Complex Aggregation ──────────────────────────────────────────────

class HardComplexAggregationGrader(BaseGrader):
    """
    Grades solutions to the complex aggregation task.
    Perfect solution: CTE + LAG() window function + date range instead of YEAR()/MONTH().
    """

    def grade(self, original_query: str, optimized_query: str) -> GradeResult:
        explanation_parts = []

        if not self._syntax_ok(optimized_query):
            return GradeResult(
                score=0.0, correctness=0.0, performance=0.0, quality=0.0,
                explanation="Syntax error: could not find SELECT ... FROM structure."
            )

        norm = self._normalize(optimized_query)

        # Correctness: still aggregates sales with category + month breakdown
        has_sales = self._has_pattern(norm, r"\bsales\b")
        has_products = self._has_pattern(norm, r"\bproducts\b")
        has_category = self._has_pattern(norm, r"\bcategory\b")
        has_aggregation = self._has_pattern(norm, r"\bsum\s*\(")
        has_revenue = self._has_pattern(norm, r"revenue|unit_price|amount")

        correctness = 0.0
        if has_sales:
            correctness += 0.2
            explanation_parts.append("✓ References sales table")
        if has_products:
            correctness += 0.2
            explanation_parts.append("✓ References products table")
        if has_category:
            correctness += 0.2
            explanation_parts.append("✓ Includes category grouping")
        if has_aggregation:
            correctness += 0.2
            explanation_parts.append("✓ Uses SUM() aggregation")
        if has_revenue:
            correctness += 0.2
            explanation_parts.append("✓ Computes revenue correctly")

        # Performance scoring
        performance = 0.0

        # Key optimization 1: Use CTE or subquery instead of correlated subquery
        has_cte = self._has_pattern(norm, r"\bwith\b.+\bas\s*\(")
        has_lag = self._has_pattern(norm, r"\blag\s*\(")
        has_window = self._has_pattern(norm, r"\bover\s*\(")

        # Key optimization 2: SARGable date filter
        uses_non_sargable = self._has_pattern(norm, r"\byear\s*\(.+order_date")
        uses_date_range = self._has_pattern(
            norm,
            r"order_date\s*(>=|between|>)\s*['\"]?2024",
        )

        # Key optimization 3: No correlated subquery
        has_correlated = self._has_pattern(
            norm,
            r"select.+from\s+sales.+where.+category\s*=\s*p\.",
        )

        if has_cte:
            performance += 0.25
            explanation_parts.append("✓ Uses CTE (+0.25)")
        if has_lag:
            performance += 0.30
            explanation_parts.append("✓ Uses LAG() window function (+0.30)")
        if has_window:
            performance += 0.15
            explanation_parts.append("✓ Uses OVER() window (+0.15)")
        if uses_date_range and not uses_non_sargable:
            performance += 0.20
            explanation_parts.append("✓ SARGable date range filter (+0.20)")
        elif uses_non_sargable:
            explanation_parts.append("✗ Still uses non-SARGable YEAR()/MONTH() (-0.10)")
            performance = max(0.0, performance - 0.10)
        if has_correlated:
            performance = max(0.0, performance - 0.30)
            explanation_parts.append("✗ Still has correlated subquery (-0.30)")

        performance = min(1.0, performance)

        quality = self._quality_score(optimized_query)
        # Bonus quality for readable CTE structure
        if has_cte and has_lag:
            quality = min(1.0, quality + 0.1)

        total = 0.35 * correctness + 0.50 * performance + 0.15 * quality
        total = round(min(1.0, max(0.0, total)), 4)

        return GradeResult(
            score=total,
            correctness=round(correctness, 4),
            performance=round(performance, 4),
            quality=round(quality, 4),
            explanation=" | ".join(explanation_parts),
        )


# ── Grader Registry ────────────────────────────────────────────────────────

GRADERS: dict[str, BaseGrader] = {
    "easy_missing_index": EasyMissingIndexGrader(),
    "medium_n_plus_one": MediumNPlusOneGrader(),
    "hard_complex_aggregation": HardComplexAggregationGrader(),
}


def get_grader(task_id: str) -> BaseGrader:
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id!r}. Available: {list(GRADERS)}")
    return GRADERS[task_id]
