import argparse
import importlib
import sys
import os
import dotenv
import json  # For pretty printing results
import asyncio  # Add asyncio


def main():
    # Load environment variables from .env file at the project root
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Agent Automator CLI: Run AI agents.")

    # Define subparsers for commands (currently only 'run')
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # --- 'run' command ---
    parser_run = subparsers.add_parser("run", help="Run a specific agent")
    parser_run.add_argument(
        "agent_name",
        type=str,
        help="Name of the agent to run (e.g., 'manim_agent').",
    )
    parser_run.add_argument(
        "script_segment",
        type=str,
        help="The script segment or main input text for the agent.",
    )
    parser_run.add_argument(
        "--script-context",
        type=str,
        help="Optional full script context to provide additional background.",
        default=None,
    )
    parser_run.add_argument(
        "--input-metadata",
        type=str,
        help="Optional metadata string for the agent (e.g., 'Key:Value,Key2:Val2').",
        default=None,
    )

    args = parser.parse_args()

    if args.command == "run":
        agent_name = args.agent_name
        runner_module_path = f"agents.{agent_name}.runner"

        print(f"Attempting to run agent: {agent_name} using runner: {runner_module_path}")

        # Remove old subprocess debug prints

        try:
            runner_module = importlib.import_module(runner_module_path)
        except ModuleNotFoundError:
            print(f"Error: Runner module '{runner_module_path}' not found.")
            print(
                f"Ensure a file named 'runner.py' exists in the 'agents/{agent_name}/' directory."
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error importing runner module '{runner_module_path}': {e}")
            sys.exit(1)

        if not (hasattr(runner_module, "execute") and callable(runner_module.execute)):
            print(
                f"Error: Runner module '{runner_module_path}' does not have a callable 'execute' function."
            )
            sys.exit(1)

        try:
            print("--- Executing Agent via CLI ---")
            # Use asyncio.run() to execute the async function from sync context
            result_dict = asyncio.run(
                runner_module.execute(
                    script_segment=args.script_segment,
                    script_context=args.script_context,
                    input_metadata=args.input_metadata,
                )
            )
            print("--- Agent Execution Finished ---")

            # Pretty print the results
            print("\n--- Execution Results ---")
            # Exclude the potentially large 'final_state' from default CLI output
            cli_result = {k: v for k, v in result_dict.items() if k != "final_state"}
            print(json.dumps(cli_result, indent=2))

            if not result_dict.get("success", False):
                sys.exit(1)  # Exit with error code if agent failed

        except Exception as e:
            print(f"\n--- Error During Agent Execution --- ")
            print(f"An unexpected error occurred: {e}")
            # Include traceback for detailed debugging
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
