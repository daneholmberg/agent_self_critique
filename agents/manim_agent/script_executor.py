import os
import subprocess
import uuid
from typing import Dict

from config import base_config as base_cfg
from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState


class ManimScriptExecutor:
    """Handles the execution of generated Manim scripts and validates output."""

    def execute_manim_script(self, state: GraphState) -> Dict:
        """Executes the generated Manim script and validates the output video."""
        print("---EXECUTE MANIM SCRIPT NODE---")
        updates_to_state: Dict = {}
        # Make copies of history lists to avoid modifying the original state directly
        error_history = state.get("error_history", [])[:]
        # Evaluation history is not directly modified here, but passed through
        evaluation_history = state.get("evaluation_history", [])[:]
        iteration = state.get("iteration", "N/A")  # For logging

        updates_to_state["error_history"] = error_history  # Initialize with copy
        updates_to_state["evaluation_history"] = evaluation_history  # Pass through
        updates_to_state["validated_artifact_path"] = None  # Default to None
        updates_to_state["validation_error"] = None  # Default to None

        # 1. Check Input Code
        manim_code = state.get("generated_output")
        if not manim_code:
            error_message = "No Manim code provided to execute."
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            error_history.append(f"Iter {iteration}: {error_message}")
            # updates_to_state already contains the updated error_history
            return updates_to_state

        temp_script_path = None  # Initialize to None
        try:
            # 2. Prepare Script File
            os.makedirs(agent_cfg.TEMP_SCRIPT_DIR, exist_ok=True)
            unique_id = uuid.uuid4()
            temp_script_filename = f"manim_script_{unique_id}.py"
            temp_script_path = os.path.join(agent_cfg.TEMP_SCRIPT_DIR, temp_script_filename)

            print(f"Writing Manim script to: {temp_script_path}")
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(manim_code)

            # 3. Construct Manim Command
            # Note: We run manim *from the workspace root* and tell it to output there.
            command = [
                "python",
                "-m",
                "manim",
                agent_cfg.MANIM_QUALITY_FLAG,
                temp_script_path,
                agent_cfg.GENERATED_SCENE_NAME,
                "--output_dir",
                ".",  # Output relative to current working directory (workspace root)
            ]
            print(f"Running Manim command: {' '.join(command)}")

            # 4. Execute Subprocess
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=base_cfg.BASE_DIR,  # Ensure we run from the project root
            )

            # 5. Check Result
            if result.returncode != 0:
                # Combine stdout and stderr for more context in errors
                full_error_output = f"Stderr:\n{result.stderr}\nStdout:\n{result.stdout}"
                error_message = f"Manim execution failed with return code {result.returncode}.\n{full_error_output}"
                print(
                    f"ERROR: Manim execution failed (code: {result.returncode}). Check logs/stderr."
                )
                # Store a concise error message in state, full details in history
                updates_to_state["validation_error"] = (
                    f"Manim execution failed with return code {result.returncode}. See history for details."
                )
                error_history.append(f"Iter {iteration}: Manim Execution Error:\n{error_message}")
                # validated_artifact_path remains None
            else:
                print("Manim execution successful (return code 0).")
                # Construct expected path relative to the workspace root
                expected_relative_path = os.path.join(
                    agent_cfg.MANIM_VIDEO_OUTPUT_RELATIVE_PATH, agent_cfg.EXPECTED_VIDEO_FILENAME
                )
                expected_full_path = os.path.join(base_cfg.BASE_DIR, expected_relative_path)

                print(f"Checking for video artifact at: {expected_full_path}")
                if os.path.exists(expected_full_path):
                    print("Video artifact found.")
                    updates_to_state["validated_artifact_path"] = (
                        expected_relative_path  # Store relative path
                    )
                    updates_to_state["validation_error"] = None  # Clear validation error
                else:
                    error_message = f"Manim reported success, but expected video not found at: {expected_full_path}"
                    print(f"ERROR: {error_message}")
                    updates_to_state["validation_error"] = error_message
                    error_history.append(f"Iter {iteration}: {error_message}")
                    # validated_artifact_path remains None

        except FileNotFoundError as e:
            # This likely means the temp script directory couldn't be created or written to
            error_message = f"Error writing temporary script: {e}"
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            error_history.append(f"Iter {iteration}: File Error: {error_message}")
            # validated_artifact_path remains None
        except Exception as e:  # Catch other potential errors during execution
            error_message = f"An unexpected error occurred during Manim script execution: {e}"
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            error_history.append(f"Iter {iteration}: Unexpected Execution Error: {error_message}")
            # validated_artifact_path remains None
        finally:
            # 6. Cleanup Temp Script (Optional but recommended)
            if temp_script_path and os.path.exists(temp_script_path):
                try:
                    os.remove(temp_script_path)
                    print(f"Cleaned up temporary script: {temp_script_path}")
                except OSError as e:
                    print(f"Warning: Could not delete temporary script {temp_script_path}: {e}")

        # 7. Return updated state fields
        return updates_to_state
