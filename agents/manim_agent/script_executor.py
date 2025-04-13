import os
import asyncio
import subprocess
import uuid
import re
import traceback
from typing import Dict, Any
from pathlib import Path

from config import base_config as base_cfg
from agents.manim_agent import config as agent_cfg
from agents.manim_agent.config import ManimAgentState
from core.log_utils import log_run_details


class ManimScriptExecutor:
    """Handles the execution of generated Manim scripts and validates output."""

    async def execute_manim_script(self, state: ManimAgentState) -> Dict[str, Any]:
        """Executes the generated Manim script asynchronously and validates the output video."""
        node_name = "ScriptExecutor"
        run_output_dir = Path(state.get("run_output_dir", "outputs/default_run"))
        current_attempt = state.get("attempt_number", 0) + 1
        scene_name = state.get("scene_name", agent_cfg.GENERATED_SCENE_NAME)

        log_run_details(
            run_output_dir, current_attempt, node_name, "Node Entry", f"Starting {node_name}..."
        )

        # --- Get info from state ---
        save_generated_code_flag = state.get("save_generated_code", False)
        manim_code = state.get("code")

        updates_to_state: Dict[str, Any] = {
            "validation_error": None,
            "execution_success": False,
            "script_file_path": None,
            "video_path": None,
            "run_output_dir": str(run_output_dir),
            "scene_name": scene_name,
            "save_generated_code": save_generated_code_flag,
        }

        # 1. Check Input Code
        if not manim_code:
            error_message = "No Manim code provided to execute (likely generator error)."
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Input Error",
                error_message,
                is_error=True,
            )
            updates_to_state["validation_error"] = error_message
            return updates_to_state

        temp_script_path: Path = None
        saved_script_path: Path = None
        try:
            # 2. Prepare Script File(s)
            temp_script_dir = run_output_dir / "temp_scripts"
            temp_script_dir.mkdir(parents=True, exist_ok=True)
            unique_id = uuid.uuid4()
            temp_script_filename = f"manim_script_iter_{current_attempt}_{unique_id}.py"
            temp_script_path = temp_script_dir / temp_script_filename
            updates_to_state["script_file_path"] = str(temp_script_path)
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "File Setup",
                f"Creating temporary script: {temp_script_path}",
            )

            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(manim_code)

            if save_generated_code_flag:
                saved_code_dir = run_output_dir / "saved_generated_code"
                saved_code_dir.mkdir(parents=True, exist_ok=True)
                saved_script_filename = f"generated_code_iter_{current_attempt}.py"
                saved_script_path = saved_code_dir / saved_script_filename
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "File Setup",
                    f"Saving copy of generated code to: {saved_script_path}",
                )
                with open(saved_script_path, "w", encoding="utf-8") as f:
                    f.write(manim_code)

            manim_output_subdir_name = "manim_media"
            manim_output_dir_in_run = run_output_dir / manim_output_subdir_name
            relative_manim_output_dir = manim_output_dir_in_run.relative_to(base_cfg.BASE_DIR)

            relative_script_path = temp_script_path.relative_to(base_cfg.BASE_DIR)

            quality_flag_to_dir = {
                "-ql": "480p15",  # Low quality
                "-qm": "720p30",  # Medium quality
                "-qh": "1080p60",  # High quality
                "-qk": "2160p60",  # 4K quality
            }
            quality_dir = quality_flag_to_dir.get(agent_cfg.MANIM_QUALITY_FLAG, "480p15")

            safe_scene_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", scene_name)
            scene_filename_mp4 = f"{safe_scene_name}.mp4"

            expected_video_full_path = (
                manim_output_dir_in_run
                / "videos"
                / temp_script_path.stem
                / quality_dir
                / scene_filename_mp4
            )

            command = [
                "python",
                "-m",
                "manim",
                str(relative_script_path),
                scene_name,
                agent_cfg.MANIM_QUALITY_FLAG,
                "--media_dir",
                str(relative_manim_output_dir),
                "-v",
                "INFO",
            ]
            command_str = " ".join(map(str, command))
            log_run_details(
                run_output_dir,
                current_attempt,
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

            stdout_decoded = ""
            stderr_decoded = ""
            return_code = -1
            timeout_occurred = False
            timeout_seconds = agent_cfg.MANIM_EXECUTION_TIMEOUT_SECONDS

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )
                return_code = process.returncode
                stdout_decoded = stdout_bytes.decode("utf-8", errors="replace").strip()
                stderr_decoded = stderr_bytes.decode("utf-8", errors="replace").strip()

                # Log full output to separate files for easier debugging
                stdout_log_path = run_output_dir / f"manim_stdout_attempt_{current_attempt}.log"
                stderr_log_path = run_output_dir / f"manim_stderr_attempt_{current_attempt}.log"
                with open(stdout_log_path, "w", encoding="utf-8") as f_out:
                    f_out.write(stdout_decoded)
                with open(stderr_log_path, "w", encoding="utf-8") as f_err:
                    f_err.write(stderr_decoded)

                # Log summary to run_details
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Process Output",
                    f"Return Code: {return_code}\nStdout Log: {stdout_log_path}\nStderr Log: {stderr_log_path}",
                )

            except asyncio.TimeoutError:
                timeout_occurred = True
                error_message = f"Manim execution timed out after {timeout_seconds} seconds. Process terminated."
                print(f"ERROR: {error_message}")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Timeout Error",
                    error_message,
                    is_error=True,
                )
                try:
                    process.kill()
                    await process.wait()
                    print("Manim process killed due to timeout.")
                except ProcessLookupError:
                    print("Manim process already terminated before kill attempt.")
                except Exception as kill_err:
                    print(f"Error trying to kill timed-out Manim process: {kill_err}")

                updates_to_state["validation_error"] = error_message

            except Exception as comm_err:
                error_message = f"Error during Manim process communication: {comm_err}"
                print(f"ERROR: {error_message}")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Communication Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["validation_error"] = error_message
                if process.returncode is None:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass

            full_process_output = f"Return Code: {return_code}\n--- Stdout ---\n{stdout_decoded}\n--- Stderr ---\n{stderr_decoded}"
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Command Result",
                full_process_output,
                is_error=(return_code != 0),
            )

            if not timeout_occurred and return_code == 0:
                print(f"Manim execution successful (code: {return_code}).")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Execution Status",
                    "Manim process completed successfully.",
                )

                if expected_video_full_path.is_file():
                    print(f"Validated video output exists: {expected_video_full_path}")
                    updates_to_state["execution_success"] = True
                    updates_to_state["validation_error"] = None
                    updates_to_state["video_path"] = str(expected_video_full_path)
                    log_run_details(
                        run_output_dir,
                        current_attempt,
                        node_name,
                        "Output Validation",
                        f"Video file found: {expected_video_full_path}",
                    )
                else:
                    error_message = f"Manim process succeeded but expected video file not found at: {expected_video_full_path}"
                    print(f"ERROR: {error_message}")
                    log_run_details(
                        run_output_dir,
                        current_attempt,
                        node_name,
                        "Output Error",
                        error_message,
                        is_error=True,
                    )
                    updates_to_state["validation_error"] = error_message

            elif not timeout_occurred and return_code != 0:
                full_error_output = f"Stderr:\n{stderr_decoded}\nStdout:\n{stdout_decoded}"
                print(f"ERROR: Manim execution failed (code: {return_code}).")
                cmd_line_error_pattern = r"(Usage:|Error: No such option|Error: Invalid argument|Error: unrecognized arguments)"
                is_cmd_line_error = bool(
                    re.search(cmd_line_error_pattern, stderr_decoded, re.IGNORECASE)
                )

                if is_cmd_line_error:
                    error_message = f"Manim command failed due to likely argument error (code {return_code}). Stderr: {stderr_decoded[:500]}..."
                    print(
                        "Detected command-line argument error. Classifying as infrastructure/config error."
                    )
                    log_run_details(
                        run_output_dir,
                        current_attempt,
                        node_name,
                        "Command Error",
                        error_message,
                        is_error=True,
                    )
                    updates_to_state["validation_error"] = f"[Config/Command Error] {error_message}"
                else:
                    error_message = f"Manim script execution failed (code {return_code}). Stderr: {stderr_decoded[:1000]}..."
                    print("Detected likely script error.")
                    log_run_details(
                        run_output_dir,
                        current_attempt,
                        node_name,
                        "Script Error",
                        error_message,
                        is_error=True,
                    )
                    updates_to_state["validation_error"] = (
                        f"Return Code: {return_code}\n{full_error_output}"
                    )

                updates_to_state["execution_success"] = False

        except Exception as e:
            error_message = (
                f"Unexpected error in ManimScriptExecutor: {e}\n{traceback.format_exc()}"
            )
            print(f"FATAL ERROR in executor: {error_message}")
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Executor Error",
                error_message,
                is_error=True,
            )
            updates_to_state["validation_error"] = f"[Executor Internal Error] {error_message}"
            updates_to_state["execution_success"] = False

        finally:
            if temp_script_path and temp_script_path.exists():
                try:
                    os.remove(temp_script_path)
                    log_run_details(
                        run_output_dir,
                        current_attempt,
                        node_name,
                        "Cleanup",
                        f"Removed temporary script: {temp_script_path}",
                    )
                except OSError as e:
                    log_run_details(
                        run_output_dir,
                        current_attempt,
                        node_name,
                        "Cleanup Warning",
                        f"Failed to remove temporary script {temp_script_path}: {e}",
                        is_error=True,
                    )

        log_run_details(
            run_output_dir,
            current_attempt,
            node_name,
            "Node Completion",
            "Finished execution attempt.",
        )
        return updates_to_state
