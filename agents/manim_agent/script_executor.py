import os
import asyncio
import subprocess
import uuid
import re
import traceback
from typing import Dict
from pathlib import Path

from config import base_config as base_cfg
from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState
from core.log_utils import log_run_details


class ManimScriptExecutor:
    """Handles the execution of generated Manim scripts and validates output."""

    async def execute_manim_script(self, state: GraphState) -> Dict:
        """Executes the generated Manim script asynchronously and validates the output video."""
        node_name = "ScriptExecutor"
        run_output_dir = Path(state["run_output_dir"])
        iteration = state["iteration"]
        log_run_details(
            run_output_dir, iteration, node_name, "Node Entry", f"Starting {node_name}..."
        )

        # --- Get info from state ---
        save_generated_code_flag = state["save_generated_code"]
        manim_code = state.get("generated_output")
        error_history = state.get("error_history", [])[:]
        evaluation_history = state.get("evaluation_history", [])[:]
        # --- End state info ---

        updates_to_state: Dict = {
            "error_history": error_history,
            "evaluation_history": evaluation_history,
            "validated_artifact_path": None,
            "validation_error": None,
            "infrastructure_error": None,
        }

        # 1. Check Input Code
        if not manim_code:
            error_message = "No Manim code provided to execute."
            log_run_details(
                run_output_dir, iteration, node_name, "Input Error", error_message, is_error=True
            )
            updates_to_state["validation_error"] = error_message
            error_history.append(f"Iter {iteration}: {error_message}")
            return updates_to_state

        temp_script_path = None
        saved_script_path = None
        try:
            # 2. Prepare Script File(s)
            temp_script_dir = run_output_dir / "temp_scripts"
            temp_script_dir.mkdir(parents=True, exist_ok=True)
            unique_id = uuid.uuid4()
            temp_script_filename = f"manim_script_{iteration}_{unique_id}.py"
            temp_script_path = temp_script_dir / temp_script_filename
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "File Setup",
                f"Creating temporary script: {temp_script_path}",
            )

            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(manim_code)

            if save_generated_code_flag:
                saved_code_dir = run_output_dir / "saved_generated_code"
                saved_code_dir.mkdir(parents=True, exist_ok=True)
                saved_script_filename = f"generated_code_iter_{iteration}.py"
                saved_script_path = saved_code_dir / saved_script_filename
                log_run_details(
                    run_output_dir,
                    iteration,
                    node_name,
                    "File Setup",
                    f"Saving copy of generated code to: {saved_script_path}",
                )
                with open(saved_script_path, "w", encoding="utf-8") as f:
                    f.write(manim_code)

            manim_output_subdir_name = "manim_media"
            manim_output_dir_in_run = run_output_dir / manim_output_subdir_name
            relative_manim_output_dir = manim_output_dir_in_run.relative_to(base_cfg.BASE_DIR)

            # Construct path considering the temp script name subdirectory
            temp_script_stem = temp_script_path.stem

            # Determine quality directory based on the flag used in the command
            quality_flag_to_dir = {
                "-ql": "480p15",  # Low quality
                "-qm": "720p30",  # Medium quality
                "-qh": "1080p60",  # High quality
                "-qk": "2160p60",  # 4K quality
            }
            # Use the mapping, default to low quality if flag unknown (should match command)
            quality_dir = quality_flag_to_dir.get(agent_cfg.MANIM_QUALITY_FLAG, "480p15")

            # Construct filename from the configured scene name
            scene_filename = f"{agent_cfg.GENERATED_SCENE_NAME}.mp4"

            expected_video_full_path = (
                run_output_dir
                / "manim_media"
                / "videos"  # Base video directory
                / temp_script_stem  # Subdirectory named after the script file
                / quality_dir  # e.g., 480p15
                / scene_filename  # e.g., GeneratedScene.mp4
            )
            expected_video_path_in_run = expected_video_full_path.relative_to(run_output_dir)

            command = [
                "python",
                "-m",
                "manim",
                str(temp_script_path.relative_to(base_cfg.BASE_DIR)),
                agent_cfg.GENERATED_SCENE_NAME,
                agent_cfg.MANIM_QUALITY_FLAG,
                "--media_dir",
                str(relative_manim_output_dir),
                "-v",
                "INFO",
            ]
            command_str = " ".join(map(str, command))
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Command Execution",
                f"Running Manim: {command_str}\nCWD: {base_cfg.BASE_DIR}",
            )

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=base_cfg.BASE_DIR,
            )

            # --- MODIFICATION: Add Timeout ---
            stdout_decoded = ""
            stderr_decoded = ""
            return_code = -1  # Default to indicate potential timeout or other issue
            timeout_occurred = False

            try:
                # Wait for the process to complete with a timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=agent_cfg.MANIM_EXECUTION_TIMEOUT_SECONDS
                )
                stdout_decoded = stdout.decode().strip() if stdout else ""
                stderr_decoded = stderr.decode().strip() if stderr else ""
                return_code = process.returncode

            except asyncio.TimeoutError:
                timeout_occurred = True
                error_message = f"Manim execution timed out after {agent_cfg.MANIM_EXECUTION_TIMEOUT_SECONDS} seconds. Process terminated."
                print(f"ERROR: {error_message}")
                log_run_details(
                    run_output_dir,
                    iteration,
                    node_name,
                    "Timeout Error",
                    error_message,
                    is_error=True,
                )
                # Attempt to kill the process
                try:
                    process.kill()
                    await process.wait()  # Ensure it's actually killed
                    print("Manim process killed due to timeout.")
                except ProcessLookupError:
                    print("Manim process already terminated before kill attempt.")
                except Exception as kill_err:
                    print(f"Error trying to kill timed-out Manim process: {kill_err}")

                # Treat timeout as a validation error for retry purposes
                updates_to_state["validation_error"] = error_message
                error_history.append(f"Iter {iteration}: Manim Timeout Error: {error_message}")
                # No artifact expected on timeout
                updates_to_state["validated_artifact_path"] = None

            except Exception as comm_err:
                # Handle other potential errors during communicate()
                error_message = f"Error during Manim process communication: {comm_err}"
                print(f"ERROR: {error_message}")
                log_run_details(
                    run_output_dir,
                    iteration,
                    node_name,
                    "Communication Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["validation_error"] = error_message  # Treat as validation error
                error_history.append(f"Iter {iteration}: Manim Comm Error: {error_message}")
                # Attempt to ensure process is cleaned up if possible
                if process.returncode is None:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass  # Ignore errors during cleanup after communication failure

            # --- End MODIFICATION ---

            # Log the final result (even after timeout/error, log what we got)
            full_process_output = f"Return Code: {return_code}\\n--- Stdout ---\\n{stdout_decoded}\\n--- Stderr ---\\n{stderr_decoded}"
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Command Result",
                full_process_output,
                is_error=(return_code != 0),  # Use the captured return code
            )

            # Only proceed with regular error/success check if no timeout occurred
            if not timeout_occurred and return_code is not None and return_code != -1:
                # --- Start INDENT ---
                if return_code != 0:
                    full_error_output = f"Stderr:\\n{stderr_decoded}\\nStdout:\\n{stdout_decoded}"
                    print(
                        f"ERROR: Manim execution failed (code: {return_code}). Check run_details.log in {run_output_dir}"
                    )
                    cmd_line_error_pattern = r"(Usage:|Error: No such option|Error: Invalid argument|Error: unrecognized arguments)"
                    is_cmd_line_error = bool(
                        re.search(cmd_line_error_pattern, stderr_decoded, re.IGNORECASE)
                    )
                    if is_cmd_line_error:
                        print(
                            "Detected command-line argument error. Classifying as infrastructure error."
                        )
                        infrastructure_error_details = f"Manim command failed due to argument error (code {return_code}). Check executor code.\\n{full_error_output}"
                        log_run_details(
                            run_output_dir,
                            iteration,
                            node_name,
                            "Command Error",
                            f"{infrastructure_error_details}\\n{full_process_output}",
                            is_error=True,
                        )
                        updates_to_state["infrastructure_error"] = infrastructure_error_details
                        updates_to_state["validation_error"] = None
                        error_history.append(
                            f"Iter {iteration}: INFRASTRUCTURE CMD ERROR: {infrastructure_error_details}"
                        )
                    else:
                        print(
                            "Manim execution failed. Classifying as validation error for LLM retry."
                        )
                        # Extract and truncate traceback from stderr for better LLM context
                        tb_str = stderr_decoded.strip()
                        tb_lines = tb_str.split("\\n")
                        max_tb_lines = 20  # Increase lines slightly for more context
                        if len(tb_lines) > max_tb_lines:
                            # Keep the first few lines and the last few lines
                            tb_summary = "\\n".join(
                                tb_lines[: max_tb_lines // 2]
                                + ["... (full traceback truncated) ..."]
                                + tb_lines[-(max_tb_lines // 2) :]
                            )
                        else:
                            tb_summary = tb_str

                        concise_error_for_llm = f"Manim execution failed (code {return_code}). Error details:\\n{tb_summary}"
                        log_run_details(
                            run_output_dir,
                            iteration,
                            node_name,
                            "Execution Error",
                            f"{concise_error_for_llm}\\n{full_process_output}",
                            is_error=True,
                        )
                        updates_to_state["validation_error"] = concise_error_for_llm
                        error_history.append(
                            f"Iter {iteration}: Manim Execution Error:\\n{concise_error_for_llm}"  # Keep concise error in history
                        )
                else:
                    print("Manim execution successful (return code 0).")
                    log_run_details(
                        run_output_dir,
                        iteration,
                        node_name,
                        "Artifact Check",
                        f"Checking for video artifact at: {expected_video_full_path}",
                    )
                    if expected_video_full_path.exists():
                        log_run_details(
                            run_output_dir,
                            iteration,
                            node_name,
                            "Artifact Found",
                            f"Video artifact found: {expected_video_path_in_run}",
                        )
                        print(
                            f"Video artifact found at {expected_video_full_path}. Path stored relative to run dir: {expected_video_path_in_run}"
                        )
                        updates_to_state["validated_artifact_path"] = str(
                            expected_video_path_in_run
                        )
                        updates_to_state["validation_error"] = None
                    else:
                        error_message = f"Manim reported success, but expected video not found at: {expected_video_full_path}\\nStdout:\\n{stdout_decoded}\\nStderr:\\n{stderr_decoded}"
                        print(f"ERROR: {error_message}")
                        log_run_details(
                            run_output_dir,
                            iteration,
                            node_name,
                            "Missing Artifact Error",
                            error_message,
                            is_error=True,
                        )
                        updates_to_state["validation_error"] = error_message
                        error_history.append(
                            f"Iter {iteration}: Missing Video Error: {error_message}"
                        )
                # --- End INDENT ---

        except FileNotFoundError as e:
            error_message = f"Error related to file/directory setup: {e}"
            print(
                f"ERROR: Infrastructure file/directory setup error: {e}. Check run_details.log in {run_output_dir}"
            )
            tb_str = traceback.format_exc()
            # Limit traceback length for clarity
            tb_lines = tb_str.strip().split("\n")
            max_tb_lines = 15
            if len(tb_lines) > max_tb_lines:
                tb_summary = "\n".join(
                    tb_lines[-(max_tb_lines // 2) :]
                    + ["... (truncated) ..."]
                    + tb_lines[-(max_tb_lines // 2) :]
                )
            else:
                tb_summary = tb_str

            infrastructure_error_details = (
                f"{error_message}\nTraceback (most recent call last):\n{tb_summary}"
            )
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "File System Error",
                infrastructure_error_details,
                is_error=True,
            )
            updates_to_state["infrastructure_error"] = infrastructure_error_details
            updates_to_state["validation_error"] = None
            error_history.append(
                f"Iter {iteration}: INFRASTRUCTURE FILE ERROR: {infrastructure_error_details}"
            )

        except Exception as e:
            error_message = f"An unexpected error occurred during Manim script execution: {e}"
            print(
                f"ERROR: Unexpected infrastructure error during script execution: {e}. Check run_details.log in {run_output_dir}"
            )
            tb_str = traceback.format_exc()
            # Limit traceback length for clarity
            tb_lines = tb_str.strip().split("\n")
            max_tb_lines = 15
            if len(tb_lines) > max_tb_lines:
                tb_summary = "\n".join(
                    tb_lines[-(max_tb_lines // 2) :]
                    + ["... (truncated) ..."]
                    + tb_lines[-(max_tb_lines // 2) :]
                )
            else:
                tb_summary = tb_str

            infrastructure_error_details = (
                f"{error_message}\nTraceback (most recent call last):\n{tb_summary}"
            )
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Unexpected Error",
                infrastructure_error_details,
                is_error=True,
            )
            updates_to_state["infrastructure_error"] = infrastructure_error_details
            updates_to_state["validation_error"] = None
            error_history.append(
                f"Iter {iteration}: INFRASTRUCTURE ERROR: {infrastructure_error_details}"
            )

        finally:
            if temp_script_path and temp_script_path.exists():
                try:
                    os.remove(temp_script_path)
                    log_run_details(
                        run_output_dir,
                        iteration,
                        node_name,
                        "Cleanup",
                        f"Deleted temporary script: {temp_script_path}",
                    )
                except OSError as e:
                    cleanup_error_msg = (
                        f"Warning: Could not delete temporary script {temp_script_path}: {e}"
                    )
                    log_run_details(
                        run_output_dir,
                        iteration,
                        node_name,
                        "Cleanup Error",
                        cleanup_error_msg,
                        is_error=True,
                    )
                    print(cleanup_error_msg)

        log_run_details(
            run_output_dir,
            iteration,
            node_name,
            "Node Completion",
            f"Finished {node_name}. Updates: { {k:v for k,v in updates_to_state.items() if k not in ['error_history', 'evaluation_history']} }",
        )
        return updates_to_state
