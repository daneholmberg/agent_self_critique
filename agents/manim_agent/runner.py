import os
import dotenv
import shutil
import datetime
import uuid
import sys
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Core components
from core.graph_state import GraphState
from core.graph_builder import build_graph
from config import base_config as base_cfg

# Manim Agent specific components
from agents.manim_agent import config as agent_cfg
from agents.manim_agent import llm_clients
from agents.manim_agent.code_generator import ManimCodeGenerator
from agents.manim_agent.script_executor import ManimScriptExecutor
from agents.manim_agent.video_evaluator import ManimVideoEvaluator
from core.log_utils import log_run_details

# --- Run Counter Helper ---
COUNTER_FILE_PATH = base_cfg.BASE_OUTPUT_DIR / "run_counter.txt"


def get_next_run_number() -> int:
    """Gets the next run number, creating/incrementing a counter file."""
    try:
        if not COUNTER_FILE_PATH.exists():
            COUNTER_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(COUNTER_FILE_PATH, "w") as f:
                f.write("0")
            return 1
        else:
            with open(COUNTER_FILE_PATH, "r+") as f:
                count = int(f.read().strip())
                next_count = count + 1
                f.seek(0)
                f.write(str(next_count))
                f.truncate()
                return next_count
    except Exception as e:
        print(f"Warning: Could not read/update run counter: {e}. Defaulting to 1.")
        return 1


def load_context_and_rubric(context_path: Path, rubric_path: Path) -> Tuple[str, str]:
    """Loads context and rubric files, raising FileNotFoundError if they don't exist."""
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            context_doc = f.read()
        print(f"Loaded context document: {context_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Context document not found at: {context_path}")

    try:
        with open(rubric_path, "r", encoding="utf-8") as f:
            rubric = f.read()
        print(f"Loaded rubric document: {rubric_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Rubric document not found at: {rubric_path}")

    return context_doc, rubric


# --- Refactored execute function (now async) ---
async def execute(
    script_segment: str,
    script_context: Optional[str] = None,
    input_metadata: Optional[str] = None,
    save_generated_code: bool = agent_cfg.SAVE_GENERATED_CODE_DEFAULT,  # Get default from config
    run_output_dir_override: Optional[str] = None,  # Allow overriding output dir
) -> Dict[str, Any]:
    """
    Main execution function for the Manim agent (async). Processes input arguments directly.

    Runs the agent's LangGraph application asynchronously.

    Args:
        script_segment: The text segment to be animated.
        script_context: Optional full script context for the generator.
        input_metadata: Optional metadata string.
        save_generated_code: Whether to save generated code iterations.
        run_output_dir_override: Optional run output directory override.

    Returns:
        A dictionary containing the execution results:
        {
            "success": bool,
            "message": str,
            "final_output_path": Optional[str],
            "final_artifact_path": Optional[str],
            "final_state": Optional[GraphState]
        }
    """
    agent_name = "manim_agent"
    print(f"Starting Async Manim Agent execution via runner.")
    result: Dict[str, Any] = {
        "success": False,
        "message": "Execution failed.",
        "final_output_path": None,
        "final_artifact_path": None,
        "final_state": None,
    }

    # 1. Basic Input Validation
    if not script_segment:
        error_message = "Error: Script segment is required."
        print(error_message)
        result["message"] = error_message
        return result  # Early exit

    prompt_text = script_segment
    full_script_context = script_context  # Already optional
    print("Received and validated input arguments.")

    # --- New: Setup Run Directory ---
    try:
        run_number = get_next_run_number()
        if run_output_dir_override:
            run_output_dir = Path(run_output_dir_override)
        else:
            run_output_dir = base_cfg.BASE_OUTPUT_DIR / f"run_{run_number:03d}"  # e.g., run_001
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Established run output directory: {run_output_dir}")
    except Exception as e:
        error_message = f"Error setting up run directory: {e}"
        print(error_message)
        result["message"] = error_message
        return result
    # --- End New ---

    # 2. Load Base Config & Environment Variables
    # dotenv.load_dotenv() is usually called at the entry point (main.py / web_launcher.py)
    try:
        api_key = base_cfg.get_gemini_api_key()
        if not api_key:
            raise ValueError("Gemini API key is missing.")
        print("Gemini API key loaded successfully.")
    except ValueError as e:
        error_message = f"Error: {e}. Ensure GOOGLE_API_KEY is set."
        print(error_message)
        result["message"] = error_message
        return result

    # 3. Instantiate Manim Agent Components
    try:
        print("Instantiating Manim agent components...")
        text_llm, eval_llm = llm_clients.get_llm_clients()
        code_generator = ManimCodeGenerator(
            llm_text_client=text_llm, script_context=full_script_context
        )
        script_executor = ManimScriptExecutor()
        video_evaluator = ManimVideoEvaluator(llm_eval_client=eval_llm)
        print("Manim agent components instantiated.")
    except Exception as e:
        error_message = f"Error during component instantiation: {e}"
        print(error_message)
        result["message"] = error_message
        return result

    # 4. Get Manim Agent Node Functions
    generate_func = code_generator.generate_manim_code
    validate_func = script_executor.execute_manim_script
    evaluate_func = video_evaluator.evaluate_manim_video
    print("Node functions retrieved.")

    # 5. Prepare Paths & Context
    try:
        context_doc, rubric = load_context_and_rubric(
            agent_cfg.CONTEXT_FILE_PATH, agent_cfg.RUBRIC_FILE_PATH
        )
    except FileNotFoundError as e:
        error_message = f"Error: {e}"
        print(error_message)
        result["message"] = error_message
        return result
    except Exception as e:
        error_message = f"Error during path/context preparation: {e}"
        print(error_message)
        result["message"] = error_message
        return result

    # 6. Build Graph
    print("Building the execution graph...")
    app = build_graph(generate_func, validate_func, evaluate_func)
    print("Graph built successfully.")

    # 7. Prepare Initial State
    initial_state = GraphState(
        input_text=prompt_text,
        input_metadata=input_metadata,
        context_doc=context_doc,
        rubric=rubric,
        max_iterations=base_cfg.MAX_ITERATIONS,
        iteration=0,
        generated_output=None,
        validation_error=None,
        validated_artifact_path=None,
        evaluation_feedback=None,
        evaluation_passed=None,
        error_history=[],
        evaluation_history=[],
        final_output_path=None,
        final_artifact_path=None,
        infrastructure_error=None,  # Initialize explicitly
        # --- New fields for state ---
        run_output_dir=str(run_output_dir),  # Pass absolute path as string
        save_generated_code=save_generated_code,
        # --- End New ---
    )
    print("Initial state prepared.")

    # 8. Invoke Graph Asynchronously
    print("\n--- Invoking Manim Agent Graph Asynchronously --- ")
    final_state = None
    try:
        # Use ainvoke for async execution
        final_state = await app.ainvoke(initial_state)
        result["final_state"] = final_state  # Store final state
        print("\n--- Manim Agent Graph Async Execution Finished ---")
    except Exception as e:
        error_message = f"An unexpected error occurred during async graph execution: {e}"
        print(f"\n--- Manim Agent Graph Async Execution FAILED --- ")
        print(error_message)
        # Include traceback
        import traceback

        traceback.print_exc()
        result["message"] = error_message
        return result  # Exit after graph execution failure

    # 9. Process Final State
    print("\n--- Processing Final State ---")
    if final_state and final_state.get("evaluation_passed"):
        print("\nSUCCESS: Manim Agent task completed and evaluation passed.")
        result["success"] = True
        result["message"] = "Agent task completed and evaluation passed."

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        # --- Adjusted: Use run_output_dir ---
        run_output_path_obj = Path(final_state["run_output_dir"])
        base_filename = f"{agent_name}_{timestamp}_{unique_id}"
        # final_output_path_obj = agent_cfg.FINAL_AGENT_OUTPUT_DIR / f"{base_filename}_code.py"
        # final_artifact_path_obj = agent_cfg.FINAL_AGENT_OUTPUT_DIR / f"{base_filename}_video.mp4"
        final_output_path_obj = run_output_path_obj / f"{base_filename}_code.py"
        final_artifact_path_obj = run_output_path_obj / f"{base_filename}_video.mp4"
        # --- End Adjustment ---

        if final_state.get("generated_output"):
            try:
                with open(final_output_path_obj, "w", encoding="utf-8") as f:
                    f.write(final_state["generated_output"])
                final_output_path_str = str(final_output_path_obj)
                print(f"Final output saved to: {final_output_path_str}")
                # Update final state in place for clarity (optional but good practice)
                final_state["final_output_path"] = final_output_path_str
                result["final_output_path"] = final_output_path_str  # Add to result dict
            except Exception as e:
                print(f"Error saving final output: {e}")
                result["message"] += f" (Warning: Error saving final output: {e})"
        else:
            print("Warning: Evaluation passed, but no final output (code) found.")
            result["message"] += " (Warning: No final output code found)"

        validated_artifact_path = final_state.get("validated_artifact_path")
        if validated_artifact_path:
            # --- Adjusted: Use run_output_dir and handle relative path carefully ---
            # source_artifact_full_path = base_cfg.PROJECT_ROOT / validated_artifact_path
            # validated_artifact_path is relative to the *run* output dir now (set by executor)
            source_artifact_full_path = run_output_path_obj / validated_artifact_path
            # --- End Adjustment ---
            if source_artifact_full_path.exists():
                try:
                    print(
                        f"Copying artifact from {source_artifact_full_path} to {final_artifact_path_obj}"
                    )
                    shutil.copy2(source_artifact_full_path, final_artifact_path_obj)
                    final_artifact_path_str = str(final_artifact_path_obj)
                    print(f"Final artifact saved to: {final_artifact_path_str}")
                    # Update final state in place
                    final_state["final_artifact_path"] = final_artifact_path_str
                    result["final_artifact_path"] = final_artifact_path_str  # Add to result dict
                    try:
                        os.remove(source_artifact_full_path)
                        print(f"Removed temporary artifact: {source_artifact_full_path}")
                    except OSError as e:
                        print(
                            f"Warning: Could not remove temp artifact {source_artifact_full_path}: {e}"
                        )
                except Exception as e:
                    print(f"Error copying/saving final artifact: {e}")
                    result["message"] += f" (Warning: Error saving final artifact: {e})"
            else:
                print(f"Warning: Artifact path {source_artifact_full_path} not found.")
                result["message"] += f" (Warning: Artifact {source_artifact_full_path} not found)"
        else:
            print("Warning: Evaluation passed, but no validated artifact path found.")
            result["message"] += " (Warning: No final artifact path found)"
    else:
        # Failure case (already printed specific reasons during graph execution)
        print("\nFAILURE: Manim Agent task did not pass evaluation or encountered errors.")
        result["success"] = False
        if final_state:
            fail_reason = "Unknown (check logs/state, possibly max iterations)."
            if not final_state.get("evaluation_passed") and final_state.get("evaluation_feedback"):
                fail_reason = "Evaluation failed."
            elif final_state.get("validation_error"):
                fail_reason = "Last step resulted in a validation error."
            result["message"] = f"Agent task failed: {fail_reason}"
            # Optionally include history in message if needed for API response
            # error_hist = final_state.get("error_history", [])
            # eval_hist = final_state.get("evaluation_history", [])
        else:
            # This case occurs if app.ainvoke itself failed catastrophically
            # The error message is already set from the exception handler
            pass

    print("\nAsync Manim Agent runner finished.")
    return result
