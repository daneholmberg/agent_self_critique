import argparse
import importlib
import sys
import os
import dotenv


def main():
    # Load environment variables from .env file at the project root
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Agent Automator: Run AI agents.")

    # Define subparsers for commands (currently only 'run')
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # --- 'run' command ---
    parser_run = subparsers.add_parser("run", help="Run a specific agent")
    parser_run.add_argument(
        "agent_name",
        type=str,
        help="Name of the agent to run (must correspond to an agent directory in 'agents/', e.g., 'manim_agent').",
    )

    # Optional metadata (remains a command-line argument)
    parser_run.add_argument(
        "--input-metadata",
        type=str,
        help="Optional metadata string for the agent (e.g., 'Key: Value, AnotherKey: Value').",
        default=None,
    )

    args = parser.parse_args()

    if args.command == "run":
        agent_name = args.agent_name

        # Construct the path to the agent's runner module
        runner_path = f"agents.{agent_name}.runner"

        print(f"Attempting to run agent: {agent_name} using runner: {runner_path}")

        try:
            # Dynamically import the specific agent runner module
            runner_module = importlib.import_module(runner_path)
        except ModuleNotFoundError:
            print(f"Error: Runner module '{runner_path}' not found.")
            print(
                f"Ensure a file named 'runner.py' exists in the 'agents/{agent_name}/' directory."
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error importing runner module '{runner_path}': {e}")
            sys.exit(1)

        # Check if the runner module has an 'execute' function
        if hasattr(runner_module, "execute") and callable(runner_module.execute):
            try:
                # Call the execute function, passing only metadata
                runner_module.execute(input_metadata=args.input_metadata)
            except Exception as e:
                print(f"Error during execution of agent '{agent_name}': {e}")
                # Potentially add more detailed error logging here
                sys.exit(1)
        else:
            print(
                f"Error: Runner module '{runner_path}' does not have a callable 'execute' function."
            )
            sys.exit(1)
    else:
        # Should not happen if subparsers.required is True
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
