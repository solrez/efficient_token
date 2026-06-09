from langchain_core.tools import tool

VERBOSE_DESCRIPTION = (
    "Execute a Python code snippet in a sandboxed environment. The code runs in an "
    "isolated Python 3.11 runtime with access to standard library modules and common "
    "data science packages (pandas, numpy, json, csv, math, statistics, collections, "
    "itertools, datetime).\n\n"
    "The code's stdout output is captured and returned. If the code raises an exception, "
    "the traceback is returned instead. Use print() statements to produce output.\n\n"
    "Use this tool for data processing, analysis scripts, format conversions, or any "
    "computation that's easier to express in Python than as a natural language description.\n\n"
    "Example: run_code(code='import json\\ndata = {\"key\": \"value\"}\\nprint(json.dumps(data, indent=2))')\n\n"
    "Limitations:\n"
    "  - No network access (no requests, urllib, etc.)\n"
    "  - No filesystem access (use file_read/file_write tools instead)\n"
    "  - 30-second execution timeout\n"
    "  - 256MB memory limit\n"
    "  - No subprocess or os.system calls"
)

RUN_CODE_PRUNED = "Execute a Python snippet. Has access to standard library and pandas/numpy. Returns stdout."

RUN_CODE_ROUTING = "Run Python code."


@tool(description=VERBOSE_DESCRIPTION)
def run_code(code: str) -> str:
    """Execute Python code in a sandboxed environment."""
    return (
        f"Code executed successfully.\n\n"
        f"Output:\n"
        f"Customer Analysis Summary\n"
        f"========================\n"
        f"Total customers: 10\n"
        f"Total revenue: $28,170,000\n"
        f"Average revenue: $2,817,000\n"
        f"Median revenue: $2,865,000\n"
        f"Std deviation: $967,432\n"
        f"Top performer: Globex Corp ($4,250,000)\n"
        f"Highest growth: Cyberdyne Systems (+31.2%)\n"
        f"At-risk (negative growth): Umbrella Inc (-2.1%), Wonka Industries (-8.4%)\n"
    )


run_code_tool = run_code
