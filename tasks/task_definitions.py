"""
Task definitions for the SQL Query Optimizer environment.
Each task has a scenario with schema, original query, execution plan, and hints.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    task_id: str
    difficulty: str  # easy | medium | hard
    description: str
    schema: str
    original_query: str
    sample_data_description: str
    query_plan: str
    performance_hint: str
    # For grader: what patterns constitute a good solution
    expected_optimizations: list[str]
    anti_patterns: list[str]


TASKS = {
    "easy_missing_index": Task(
        task_id="easy_missing_index",
        difficulty="easy",
        description=(
            "A customer lookup query is doing a full table scan on a users table "
            "with 2M rows because it filters by email but there's no index on that column. "
            "Rewrite the query AND add the appropriate index."
        ),
        schema="""
CREATE TABLE users (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    email       VARCHAR(255) NOT NULL,
    username    VARCHAR(100) NOT NULL,
    created_at  DATETIME NOT NULL DEFAULT NOW(),
    status      ENUM('active','suspended','deleted') NOT NULL DEFAULT 'active',
    plan_id     INT,
    last_login  DATETIME
);

CREATE TABLE plans (
    id          INT PRIMARY KEY,
    name        VARCHAR(50) NOT NULL,
    price_cents INT NOT NULL
);
""".strip(),
        original_query="""
SELECT u.id, u.username, u.email, p.name AS plan_name
FROM   users u
LEFT   JOIN plans p ON p.id = u.plan_id
WHERE  u.email = 'alice@example.com'
  AND  u.status = 'active';
""".strip(),
        sample_data_description=(
            "users: 2,000,000 rows. ~5% have status='active'. "
            "email values are unique. plans: 5 rows."
        ),
        query_plan="""
-> Nested loop left join  (cost=201432 rows=201432)
    -> Filter: (u.status = 'active')  (cost=201003 rows=201003)
        -> Table scan on u  (cost=201003 rows=2010033)   <-- FULL SCAN!
    -> Single-row index lookup on p using PRIMARY (id=u.plan_id)
""".strip(),
        performance_hint=(
            "The query filters by email but there is no index on users.email. "
            "Adding a unique index on email would turn the full scan into a single-row lookup."
        ),
        expected_optimizations=[
            "CREATE.*INDEX.*ON.*users.*email",
            "CREATE.*UNIQUE.*INDEX.*ON.*users.*email",
        ],
        anti_patterns=[
            "SELECT \\*",
        ],
    ),

    "medium_n_plus_one": Task(
        task_id="medium_n_plus_one",
        difficulty="medium",
        description=(
            "An application fetches all orders then fires a separate query per order "
            "to get the customer name — classic N+1 problem. The code generates "
            "thousands of small queries. Rewrite this as a single efficient JOIN query."
        ),
        schema="""
CREATE TABLE orders (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    customer_id  BIGINT NOT NULL,
    product_id   BIGINT NOT NULL,
    amount_cents INT NOT NULL,
    status       VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at   DATETIME NOT NULL DEFAULT NOW(),
    INDEX idx_customer (customer_id),
    INDEX idx_status   (status)
);

CREATE TABLE customers (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    name         VARCHAR(200) NOT NULL,
    email        VARCHAR(255) NOT NULL UNIQUE,
    country_code CHAR(2)
);

CREATE TABLE products (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    name         VARCHAR(200) NOT NULL,
    category     VARCHAR(100),
    sku          VARCHAR(50) UNIQUE
);
""".strip(),
        original_query="""
-- Application code fires this query, then for EACH order fires the customer query below:

-- Query 1: Fetch all recent orders
SELECT id, customer_id, product_id, amount_cents, status
FROM   orders
WHERE  status = 'completed'
  AND  created_at >= '2024-01-01';

-- Query 2 (fired N times, once per order):
SELECT name, email
FROM   customers
WHERE  id = ?;   -- ? = customer_id from Query 1
""".strip(),
        sample_data_description=(
            "orders: 500,000 rows, ~120,000 with status='completed' after 2024-01-01. "
            "customers: 80,000 rows. products: 10,000 rows. "
            "The N+1 pattern fires ~120,000 individual customer lookups."
        ),
        query_plan="""
-- Query 1 plan:
-> Index lookup on orders using idx_status (status='completed')
   Filter: created_at >= '2024-01-01'   (cost=12340)

-- Query 2 plan (×120,000 times):
-> Single-row index lookup on customers using PRIMARY (id=?)  (cost=1 each)
-- Total cost: 12340 + 120000 × 1 = 132340 round trips!
""".strip(),
        performance_hint=(
            "The N+1 pattern causes 120,000+ database round trips. "
            "A single JOIN between orders and customers (and optionally products) "
            "would collapse this to one query with one round trip."
        ),
        expected_optimizations=[
            "JOIN.*customers",
            "orders.*JOIN",
            "JOIN.*orders",
        ],
        anti_patterns=[
            "WHERE.*id\\s*=\\s*\\?",  # still parameterized per-row lookup
        ],
    ),

    "hard_complex_aggregation": Task(
        task_id="hard_complex_aggregation",
        difficulty="hard",
        description=(
            "A business intelligence query computes monthly revenue by category "
            "using correlated subqueries and non-SARGable predicates. It runs for "
            "45+ seconds on the reporting database. Rewrite it to run in under 2 seconds "
            "using proper aggregation, CTEs, and window functions."
        ),
        schema="""
CREATE TABLE sales (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    order_date   DATE NOT NULL,
    product_id   BIGINT NOT NULL,
    quantity     INT NOT NULL,
    unit_price   DECIMAL(10,2) NOT NULL,
    discount_pct DECIMAL(5,2) NOT NULL DEFAULT 0,
    region_id    INT NOT NULL,
    INDEX idx_date      (order_date),
    INDEX idx_product   (product_id),
    INDEX idx_region    (region_id)
);

CREATE TABLE products (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    name         VARCHAR(200) NOT NULL,
    category     VARCHAR(100) NOT NULL,
    subcategory  VARCHAR(100),
    cost_price   DECIMAL(10,2) NOT NULL,
    INDEX idx_category (category)
);

CREATE TABLE regions (
    id           INT PRIMARY KEY,
    name         VARCHAR(100) NOT NULL,
    country      VARCHAR(100) NOT NULL,
    zone         VARCHAR(50) NOT NULL
);
""".strip(),
        original_query="""
-- Current slow query (45+ seconds):
SELECT
    YEAR(s.order_date)  AS year,
    MONTH(s.order_date) AS month,
    p.category,
    SUM(s.quantity * s.unit_price * (1 - s.discount_pct/100)) AS revenue,
    (
        SELECT SUM(s2.quantity * s2.unit_price * (1 - s2.discount_pct/100))
        FROM   sales s2
        JOIN   products p2 ON p2.id = s2.product_id
        WHERE  p2.category = p.category
          AND  YEAR(s2.order_date)  = YEAR(s.order_date)
          AND  MONTH(s2.order_date) = MONTH(s.order_date) - 1
    ) AS prev_month_revenue
FROM   sales s
JOIN   products p ON p.id = s.product_id
WHERE  YEAR(s.order_date) = 2024
GROUP  BY YEAR(s.order_date), MONTH(s.order_date), p.category
ORDER  BY year, month, p.category;
""".strip(),
        sample_data_description=(
            "sales: 15,000,000 rows spanning 3 years. "
            "products: 50,000 rows across 12 categories. "
            "The correlated subquery re-scans ~5M rows per (month, category) combination. "
            "12 categories × 12 months = 144 correlated subquery executions, each scanning millions of rows."
        ),
        query_plan="""
-> Sort  (cost=99999999)
  -> Group aggregate: sum(...)
    -> Nested loop inner join
      -> Table scan on sales  (cost=1523442)  <-- FULL SCAN (YEAR() non-SARGable)
      -> Single-row index lookup on products using PRIMARY
    -> Select #2 (correlated subquery, runs 144 times):
        -> Table scan on s2  (cost=1523442 each)  <-- 144 × FULL SCAN!
        -> Join with products p2
""".strip(),
        performance_hint=(
            "Two major issues: (1) YEAR(order_date) and MONTH(order_date) predicates "
            "are non-SARGable — use a date range instead to enable index seeks. "
            "(2) The correlated subquery reruns a full scan 144 times — replace with "
            "a CTE + LAG() window function to compute prev_month_revenue in a single pass."
        ),
        expected_optimizations=[
            "WITH|CTE",
            "LAG\\(",
            "OVER\\s*\\(",
            "order_date\\s*(>=|BETWEEN)",
            "order_date\\s*>=\\s*['\"]2024",
        ],
        anti_patterns=[
            "YEAR\\(.*order_date\\)",
            "MONTH\\(.*order_date\\)",
        ],
    ),
}
