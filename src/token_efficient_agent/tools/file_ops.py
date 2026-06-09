from langchain_core.tools import tool

FILE_READ_VERBOSE = (
    "Read the contents of a file from the local filesystem. Provide the full file path "
    "as a parameter. The tool returns the complete text content of the file. Supports "
    "text files of any format (.txt, .csv, .json, .md, .py, .log, etc.). Maximum file "
    "size is 1MB. Binary files are not supported and will return an error.\n\n"
    "Use this tool to read configuration files, log files, data files, source code, "
    "or any text-based file that you need to analyze or process.\n\n"
    "Example: file_read(path='/var/log/etl/customer_events.log')\n"
    "Example: file_read(path='/tmp/config.json')\n\n"
    "Note: Paths must be absolute. Relative paths are not supported. "
    "The tool respects filesystem permissions — files you don't have read access to "
    "will return a permission error."
)

FILE_READ_PRUNED = "Read a text file from the filesystem. Provide the absolute file path."

FILE_READ_ROUTING = "Read a file."


@tool(description=FILE_READ_VERBOSE)
def file_read(path: str) -> str:
    """Read a file from the filesystem."""
    if "log" in path:
        return (
            f"Contents of {path}:\n\n"
            "[2026-06-07 14:23:01] ERROR: Pipeline customer_events failed at transform stage\n"
            "[2026-06-07 14:23:01] ERROR: KeyError: 'event_timestamp' in transform_events()\n"
            "[2026-06-07 14:23:01] DEBUG: Input schema: {user_id: str, event_type: str, ts: int, metadata: dict}\n"
            "[2026-06-07 14:23:01] DEBUG: Expected field 'event_timestamp' but found 'ts'\n"
            "[2026-06-07 14:22:58] INFO: Loaded 45,230 records from source\n"
            "[2026-06-07 14:22:55] INFO: Starting pipeline run #4521\n"
            "[2026-06-07 14:20:01] INFO: Previous run #4520 completed successfully (38,901 records)\n"
        )
    if "csv" in path or "data" in path:
        return (
            f"Contents of {path}:\n\n"
            "customer_id,name,revenue,churn_score\n"
            "C001,Globex Corp,4250000,0.12\n"
            "C002,Initech LLC,3890000,0.08\n"
            "C003,Umbrella Inc,3420000,0.45\n"
            "C004,Stark Industries,2980000,0.05\n"
            "C005,Wayne Enterprises,2750000,0.15\n"
        )
    return f"Contents of {path}:\n\n[Mock file content for {path}]"


FILE_WRITE_VERBOSE = (
    "Write content to a file on the local filesystem. Provide the full file path and "
    "the content to write. If the file exists, it will be overwritten. If the file "
    "does not exist, it will be created along with any necessary parent directories.\n\n"
    "Supports writing any text content. Use this for saving analysis results, generated "
    "reports, code files, configuration files, or any text output.\n\n"
    "Example: file_write(path='/tmp/report.md', content='# Analysis Report\\n...')\n"
    "Example: file_write(path='/tmp/data.csv', content='col1,col2\\nval1,val2')\n\n"
    "Note: Paths must be absolute. Maximum content size is 10MB. "
    "Binary content is not supported."
)

FILE_WRITE_PRUNED = "Write content to a file. Creates parent directories if needed."

FILE_WRITE_ROUTING = "Write a file."


@tool(description=FILE_WRITE_VERBOSE)
def file_write(path: str, content: str) -> str:
    """Write content to a file."""
    lines = content.count("\n") + 1
    chars = len(content)
    return f"Successfully wrote {chars} characters ({lines} lines) to {path}"


file_read_tool = file_read
file_write_tool = file_write
