"""Core utilities for logging agent run details."""

import datetime
import traceback
from pathlib import Path
from typing import Any


def log_run_details(
    run_output_dir: Path | str,
    iteration: int,
    node_name: str,
    log_category: str,
    content: Any,
    is_error: bool = False,
):
    """
    Logs detailed information to a standard file within the run output directory.

    Handles creating the directory and file if they don't exist. Appends
    timestamped, structured log entries.

    Args:
        run_output_dir: The Path object or string path to the run's output directory.
        iteration: The current iteration number of the agent run.
        node_name: The name of the node or component generating the log.
        log_category: A string categorizing the log entry (e.g., "LLM Prompt",
                      "State Update", "Node Entry", "Error").
        content: The detailed content to log (can be string, dict, etc., will be str()-ified).
        is_error: If True, indicates the log entry represents an error.
    """
    try:
        if isinstance(run_output_dir, str):
            run_output_dir = Path(run_output_dir)

        log_file_path = run_output_dir / "run_details.log"
        run_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_prefix = (
            f"[{timestamp}] Iteration={iteration} Node='{node_name}' Category='{log_category}'"
        )
        if is_error:
            log_prefix += " Status='ERROR'"

        log_entry = (
            f"{log_prefix}\n{'-'*len(log_prefix)}\n{str(content)}\n{'-'*len(log_prefix)}\n\n"
        )

        with open(log_file_path, "a", encoding="utf-8") as log_f:
            log_f.write(log_entry)

    except Exception:
        # Log the failure to stderr, but don't crash the main process
        print(f"ERROR: Failed to write agent run details to {log_file_path}:")
        traceback.print_exc()
