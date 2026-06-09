from langchain_core.tools import tool

VERBOSE_DESCRIPTION = (
    "Execute a SQL query against the company's internal database. The database contains "
    "the following tables:\n"
    "  - customers(id, name, industry, region, created_at, status)\n"
    "  - orders(id, customer_id, amount, product, created_at, status)\n"
    "  - revenue(customer_id, quarter, year, amount, growth_pct)\n"
    "  - support_tickets(id, customer_id, priority, status, created_at, resolved_at)\n\n"
    "The query parameter should be a valid SQL SELECT statement. INSERT, UPDATE, DELETE, "
    "and DDL statements are not permitted. Use standard SQL syntax. Results are returned "
    "as a formatted table with column headers. Maximum 100 rows are returned per query. "
    "For large result sets, use LIMIT and OFFSET for pagination. Joins across tables are "
    "supported. Aggregate functions (COUNT, SUM, AVG, MIN, MAX) and GROUP BY are available.\n\n"
    "Example: sql_query(query='SELECT c.name, SUM(o.amount) as total FROM customers c "
    "JOIN orders o ON c.id = o.customer_id GROUP BY c.name ORDER BY total DESC LIMIT 10')\n\n"
    "Note: Query execution timeout is 30 seconds. Complex joins on large tables may be slow."
)

SQL_QUERY_PRUNED = "Execute a SQL SELECT query against the internal database (customers, orders, revenue, support_tickets tables)."

SQL_QUERY_ROUTING = "SQL database query."


@tool(description=VERBOSE_DESCRIPTION)
def sql_query(query: str) -> str:
    """Execute a SQL query against the internal database."""
    return (
        f"Query: {query}\n\n"
        "| customer_name    | total_revenue | yoy_growth | region  | industry      |\n"
        "|------------------|---------------|------------|---------|---------------|\n"
        "| Globex Corp      | $4,250,000    | +18.2%     | NA      | Manufacturing |\n"
        "| Initech LLC      | $3,890,000    | +12.5%     | NA      | Technology    |\n"
        "| Umbrella Inc     | $3,420,000    | -2.1%      | EMEA    | Healthcare    |\n"
        "| Stark Industries | $2,980,000    | +24.7%     | NA      | Defense       |\n"
        "| Wayne Enterprises| $2,750,000    | +8.3%      | NA      | Conglomerate  |\n"
        "| Cyberdyne Systems| $2,340,000    | +31.2%     | APAC    | Technology    |\n"
        "| Soylent Corp     | $1,980,000    | +5.6%      | EMEA    | Food & Bev    |\n"
        "| Tyrell Corp      | $1,750,000    | +15.8%     | APAC    | Biotech       |\n"
        "| Wonka Industries | $1,520,000    | -8.4%      | NA      | Consumer      |\n"
        "| Oscorp           | $1,290,000    | +22.1%     | NA      | Pharma        |\n\n"
        "10 rows returned."
    )


sql_query_tool = sql_query
