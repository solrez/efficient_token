from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkScenario:
    name: str
    description: str
    initial_message: str
    expected_tools: list[str]
    min_steps: int


SCENARIOS: dict[str, BenchmarkScenario] = {
    "market_research": BenchmarkScenario(
        name="market_research",
        description="Research a company, query sales data, compute metrics, write report, email summary",
        initial_message=(
            "Research the latest quarterly performance of Acme Corp. "
            "Query our sales database for their account data, calculate YoY growth rates, "
            "write a brief analysis report to /tmp/acme_report.md, and "
            "email the summary to sales-team@company.com."
        ),
        expected_tools=["web_search", "sql_query", "calculator", "file_write", "send_email"],
        min_steps=5,
    ),
    "data_pipeline_debug": BenchmarkScenario(
        name="data_pipeline_debug",
        description="Debug a data pipeline: read logs, search docs, run diagnostic code, write fix",
        initial_message=(
            "Our ETL pipeline for customer_events is failing. Read the error log at "
            "/var/log/etl/customer_events.log, search for known issues with the "
            "error pattern you find, write and run a diagnostic Python script to "
            "validate the schema mapping, then write the fix to /tmp/pipeline_fix.py."
        ),
        expected_tools=["file_read", "web_search", "run_code", "file_write"],
        min_steps=5,
    ),
    "customer_analysis": BenchmarkScenario(
        name="customer_analysis",
        description="SQL analysis with calculations, code processing, email report",
        initial_message=(
            "Analyze our top 10 customers by revenue from the sales database. "
            "Calculate their average revenue and identify any with negative growth as at-risk. "
            "Write a Python script to generate a formatted summary table, "
            "save the results to /tmp/customer_analysis.csv, and email "
            "the highlights to management@company.com."
        ),
        expected_tools=["sql_query", "calculator", "run_code", "file_write", "send_email"],
        min_steps=5,
    ),
}
