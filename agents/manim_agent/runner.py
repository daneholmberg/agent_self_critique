import os
import dotenv
import shutil
import datetime
import uuid
import sys
import re  # Import re for filename sanitization
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

# Core components
# from core.graph_state import GraphState # Remove this if GraphState moves
from core.graph_builder import build_graph  # Keep for now, might replace later
from langgraph.graph import StateGraph, END

# from langgraph.checkpoint.sqlite import SqliteSaver # REMOVE UNUSED IMPORT
from langgraph.graph.message import add_messages
from config import base_config as base_cfg

# Manim Agent specific components
from agents.manim_agent import config as agent_cfg
from agents.manim_agent.config import ManimAgentState, DEFAULT_FAILURE_SUMMARY_PROMPT
from agents.manim_agent.code_generator import ManimCodeGenerator
from agents.manim_agent.script_executor import ManimScriptExecutor
from agents.manim_agent.video_evaluator import ManimVideoEvaluator
from agents.manim_agent.rubric_modifier import RubricModifier
from agents.manim_agent.components.failure_summarizer import FailureSummarizer
from core.log_utils import log_run_details
from core.llm.client_factory import create_llm_client

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


# --- Node Functions ---
# Moved component method calls into dedicated node functions within build_agent_graph

# --- Conditional Logic Functions ---
# (These can stay outside as they only depend on state)


# --- Graph Building Function ---
def build_agent_graph(
    code_generator: ManimCodeGenerator,
    script_executor: ManimScriptExecutor,
    video_evaluator: ManimVideoEvaluator,
    failure_summarizer: FailureSummarizer,
    rubric_modifier: Optional[RubricModifier] = None,
) -> StateGraph:
    """Builds the LangGraph StateGraph for the Manim agent."""
    print("Building Manim Agent Graph V2 with Failure Summarization...")

    # --- Define Node Functions within builder scope to access components via closure ---
    async def generate_code_node(state: ManimAgentState) -> Dict[str, Any]:  # Only takes state
        """Node to generate Manim code."""
        print("\n--- Node: Generate Code ---")
        if not state.get("task_instruction"):
            raise ValueError("Task instruction is missing from state for code generation.")
        # Use code_generator from outer scope
        result = await code_generator.generate_manim_code(state)
        # Ensure validation errors from the generator are passed through
        # Clear previous evaluation/execution states
        return {
            "code": result.get("code"),
            "previous_code_attempt": state.get("code"),
            "validation_error": result.get("validation_error"),
            "evaluation_result": None,
            "execution_success": None,
            "evaluation_passed": None,
            "video_path": None,
            "error_message": result.get("error_message"),
        }

    async def execute_script_node(state: ManimAgentState) -> Dict[str, Any]:  # Only takes state
        """Node to execute the generated Manim script."""
        print("\n--- Node: Execute Script ---")
        code = state.get("code")
        error_message = state.get("error_message")  # Error from previous generator attempt
        validation_error = state.get("validation_error")  # Error from CURRENT generator attempt

        if not code:
            # If code is missing, check if it's due to a generator error (current or previous)
            relevant_error = validation_error or error_message
            if relevant_error:
                print(
                    f"Skipping execution due to code generation/validation error: {relevant_error}"
                )
                # Indicate failure and pass the specific error message
                # Ensure execution_success is False, pass the specific error as validation_error
                # Clear error_message as it's now handled
                return {
                    "execution_success": False,
                    "validation_error": relevant_error,
                    "error_message": None,
                }
            else:
                # This case should ideally not be reached if generator always sets error/validation_error on failure
                # Log a warning before raising
                logger.warning(
                    "execute_script_node: Code is missing, but no validation_error or error_message found in state."
                )
                raise ValueError(
                    "Code is missing from state for script execution and no generator error (validation_error or error_message) found."
                )

        # If code exists, proceed with execution
        # Use script_executor from outer scope
        result = await script_executor.execute_manim_script(state)
        # Pass through results, ensure error_message is cleared if execution was attempted
        return {
            "script_file_path": result.get("script_file_path"),
            "video_path": result.get("video_path"),
            "validation_error": result.get("validation_error"),  # From execution
            "execution_success": result.get("execution_success"),
            "error_message": None,  # Clear any previous generator error if execution proceeded
        }

    async def evaluate_video_node(state: ManimAgentState) -> Dict[str, Any]:  # Only takes state
        """Node to evaluate the generated Manim video."""
        print("\n--- Node: Evaluate Video ---")
        if not state.get("video_path") or not state.get("rubric"):
            raise ValueError("Video path or rubric is missing from state for evaluation.")
        # Use video_evaluator from outer scope
        result = await video_evaluator.evaluate_manim_video(state)
        return {
            "evaluation_result": result.get("evaluation_result"),
            "evaluation_passed": result.get("evaluation_passed"),
        }

    async def modify_rubric_node(state: ManimAgentState) -> Dict[str, Any]:  # Only takes state
        """Node to potentially modify the rubric based on enhancement requests."""
        print("\n--- Node: Modify Rubric (If Enhancement Requested) ---")
        # Always pass through essential context keys
        updates = {
            "run_output_dir": state.get("run_output_dir"),
            "scene_name": state.get("scene_name"),
            "save_generated_code": state.get("save_generated_code"),
        }
        if state.get("enhancement_request") and rubric_modifier:  # Check if modifier exists
            # Use rubric_modifier from outer scope
            result = await rubric_modifier.modify_rubric_for_enhancement(state)
            updates["rubric"] = result.get("rubric")  # Update only the rubric
            print(f"Rubric potentially modified.")
            return updates
        else:
            print(
                "No enhancement request found or rubric_modifier not provided, skipping rubric modification."
            )
            # Still return the essential keys even if rubric doesn't change
            return updates

    async def summarize_single_failure_node(
        state: ManimAgentState,
    ) -> Dict[str, str]:  # Only takes state
        """Node to summarize the most recent failure (validation or evaluation)."""
        print("\n--- Node: Summarize Single Failure ---")
        # Get required info from state for logging and the call
        run_output_dir = Path(state["run_output_dir"])
        # current_attempt is the number for the attempt that *just failed* (1-based)
        # This node runs *after* a failure in attempt N (where N = attempt_number + 1)
        current_attempt = state.get("attempt_number", 0) + 1

        failure_detail = ""
        # Prioritize the failure_reason set by should_retry_or_finish
        if state.get("failure_reason"):
            failure_detail = state["failure_reason"]
            print(
                "Summarizing failure using 'failure_reason' from state (likely evaluation feedback)."
            )
        # If no failure_reason, check for validation error (code gen or execution)
        elif state.get("validation_error"):
            failure_detail = state["validation_error"]
            print("Summarizing failure using 'validation_error' from state.")
        # Fallback: Check evaluation result directly (should be covered by failure_reason now)
        elif state.get("evaluation_result") and not state.get("evaluation_passed"):
            # Extract feedback, preferring the correct key
            failure_detail = state.get("evaluation_result", {}).get(
                "feedback", "Evaluation failed - feedback key missing in result."
            )
            print(
                "Summarizing evaluation failure using direct check of 'evaluation_result.feedback'."
            )
        else:
            print(
                "Warning: summarize_single_failure_node called but no obvious failure found in state (failure_reason, validation_error, or failed evaluation_result)."
            )
            # Log this warning properly
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=current_attempt,
                node_name="SummarizeSingleFailure",  # Corrected node name capitalization
                log_category="Warning",
                content="No failure_reason, validation_error or evaluation_failure found to summarize.",
                is_error=True,
            )
            return {"single_failure_summary": "[No failure detected to summarize]"}

        if not failure_detail:
            # This case might occur if the error/feedback string itself was empty
            print("Warning: Failure detected but detail string is empty.")
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=current_attempt,
                node_name="SummarizeSingleFailure",
                log_category="Warning",
                content="Failure detected (failure_reason/validation_error/evaluation) but detail string was empty.",
                is_error=True,
            )
            return {"single_failure_summary": "[Failure detected but detail was empty]"}

        # Use failure_summarizer from outer scope
        summary = await failure_summarizer.summarize(
            failure_detail=failure_detail,
            run_output_dir=run_output_dir,
            attempt_number_failed=current_attempt,  # Use the correct keyword argument
        )
        # Clear the failure_reason after summarizing it
        return {"single_failure_summary": summary, "failure_reason": None}

    async def append_failure_summary_node(
        state: ManimAgentState,
    ) -> Dict[str, Any]:  # Only takes state
        """Node to append the latest summary, increment attempt number, and clear temporary summary field."""
        print("\n--- Node: Append Failure Summary & Increment Attempt ---")
        summary_to_append = state.get("single_failure_summary")
        current_attempt_number = state.get("attempt_number", 0)
        next_attempt_number = current_attempt_number + 1

        # Initialize updates dictionary only with fields this node modifies
        updates: Dict[str, Any] = {
            "attempt_number": next_attempt_number,
            "single_failure_summary": None,  # Always clear the temporary field
            # DO NOT CLEAR evaluation state here, it should persist for the generator
        }

        if not summary_to_append:
            print(
                "Warning: append_failure_summary_node called but no summary found in state. Only incrementing attempt number."
            )
        else:
            print(f"Appending summary and incrementing attempt number to: {next_attempt_number}")
            # Use add_messages correctly to update the list
            # Important: Get the current list or default to empty before adding
            updated_summaries = add_messages(
                state.get("failure_summaries", []), [summary_to_append]
            )
            updates["failure_summaries"] = (
                updated_summaries  # Add the updated list to the dictionary
            )

        print(f"Incrementing attempt number to: {next_attempt_number}")

        # Only return the fields that were actually updated by this node.
        # Other fields (validation_error, execution_success, etc.) remain untouched.
        return updates

    # --- End Node Function Definitions ---

    graph = StateGraph(ManimAgentState)

    # Add nodes using the inner functions directly
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_script", execute_script_node)
    graph.add_node("evaluate_video", evaluate_video_node)
    graph.add_node("summarize_failure", summarize_single_failure_node)
    graph.add_node("append_summary", append_failure_summary_node)

    if rubric_modifier:
        graph.add_node("modify_rubric", modify_rubric_node)
        graph.set_entry_point("modify_rubric")
        graph.add_edge("modify_rubric", "generate_code")
    else:
        graph.set_entry_point("generate_code")

    # Define edges and conditional logic (Conditional functions still outside)
    graph.add_edge("generate_code", "execute_script")

    graph.add_conditional_edges(
        "execute_script",
        should_retry_or_evaluate,
        {
            "evaluate_video": "evaluate_video",
            "summarize_failure": "summarize_failure",
        },
    )

    graph.add_conditional_edges(
        "evaluate_video",
        should_retry_or_finish,
        {
            END: END,
            "summarize_failure": "summarize_failure",
        },
    )

    graph.add_edge("summarize_failure", "append_summary")

    graph.add_conditional_edges(
        "append_summary", check_attempt_limit, {END: END, "generate_code": "generate_code"}
    )

    print("Manim Agent Graph V2 built.")
    return graph


# Conditional logic functions can stay outside as they only operate on state
def should_retry_or_evaluate(state: ManimAgentState) -> str:
    """Determines the next step after script execution."""
    print("--- Decision: Post-Execution Check ---")
    if state.get("execution_success"):
        print("Execution successful. Proceeding to evaluation.")
        return "evaluate_video"
    else:
        error = state.get("validation_error", "Unknown execution error")
        print(f"Execution failed: {error[:200]}... Proceeding to summarize failure.")
        return "summarize_failure"


def should_retry_or_finish(state: ManimAgentState) -> str:
    """Determines the next step after video evaluation."""
    print("--- Decision: Post-Evaluation Check ---")
    if state.get("evaluation_passed"):
        print("Evaluation successful. Finishing graph execution.")
        return END
    else:
        # Correctly extract feedback using the 'feedback' key
        evaluation_result_dict = state.get("evaluation_result", {})
        feedback = evaluation_result_dict.get(
            "feedback", "Evaluation failed: Feedback missing in state['evaluation_result']"
        )
        print(f"Evaluation failed: {feedback[:200]}... Proceeding to summarize failure.")

        # Add the extracted feedback to the state for the summarizer
        # (Assuming the summarizer expects a 'failure_reason' key)
        # NOTE: If summarizer expects a different key, update this line!
        state["failure_reason"] = feedback

        return "summarize_failure"


def check_attempt_limit(state: ManimAgentState) -> str:
    """Checks if the maximum number of attempts has been reached."""
    print("--- Decision: Attempt Limit Check ---")
    current_attempt_number = state.get("attempt_number", 0)
    max_attempts = state.get("max_attempts", 10)  # Default to 3 if not set
    if current_attempt_number >= max_attempts:
        print(
            f"Maximum attempts ({max_attempts}) reached. Ending graph execution after attempt {current_attempt_number}."
        )
        return END
    else:
        print(
            f"Attempt {current_attempt_number + 1}/{max_attempts}. Proceeding to generate code again."
        )
        return "generate_code"


# --- Refactored execute function (now async) ---
async def execute(
    script_segment: str,
    general_context: Optional[str] = None,
    previous_code_attempt: Optional[str] = None,
    enhancement_request: Optional[str] = None,
    final_command: Optional[str] = None,
    scene_name: str = agent_cfg.GENERATED_SCENE_NAME,
    save_generated_code: bool = agent_cfg.SAVE_GENERATED_CODE_DEFAULT,
    run_output_dir_override: Optional[str] = None,
    max_attempts: int = 10,  # Renamed parameter
) -> Dict[str, Any]:
    """
    Main execution function for the Manim agent (async). Processes input arguments directly.

    Runs the agent's LangGraph application asynchronously.

    Args:
        script_segment: The text segment to be animated.
        general_context: Optional general context for the generator.
        previous_code_attempt: Optional previous code to enhance.
        enhancement_request: Optional description of requested enhancements.
        final_command: Optional final command for the generator.
        scene_name: The user-provided name for the scene (used for filenames).
        save_generated_code: Whether to save generated code attempts.
        run_output_dir_override: Optional run output directory override.
        max_attempts: Maximum number of attempts for the agent.

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

    # 3. Load Context & Rubric (Moved before component instantiation needed them)
    try:
        context_doc, initial_rubric = load_context_and_rubric(
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

    # 4. Instantiate Manim Agent Components
    try:
        print("Instantiating Manim agent components...")
        text_llm = create_llm_client(
            provider=agent_cfg.LLM_PROVIDER,
            model_name=agent_cfg.TEXT_GENERATION_MODEL,
            temperature=0.9,
        )
        summarizer_llm = text_llm
        eval_llm = create_llm_client(
            provider=agent_cfg.LLM_PROVIDER,
            model_name=agent_cfg.EVALUATION_MODEL,
            temperature=0.1,
        )

        code_generator = ManimCodeGenerator(llm_text_client=text_llm)
        script_executor = ManimScriptExecutor()
        video_evaluator = ManimVideoEvaluator(llm_eval_client=eval_llm)
        rubric_modifier = RubricModifier(llm_client=text_llm) if enhancement_request else None
        failure_summarizer = FailureSummarizer(
            llm=summarizer_llm,
            summarization_prompt_template=agent_cfg.DEFAULT_FAILURE_SUMMARY_PROMPT,
        )
        print("Manim agent components instantiated.")
    except Exception as e:
        error_message = f"Error during component instantiation: {e}"
        print(error_message)
        result["message"] = error_message
        return result

    # 5. Build Graph
    print("Building the execution graph...")
    app = build_agent_graph(
        code_generator=code_generator,
        script_executor=script_executor,
        video_evaluator=video_evaluator,
        failure_summarizer=failure_summarizer,
        rubric_modifier=rubric_modifier,
    )
    app = app.compile()
    print("Graph built and compiled successfully.")

    # 6. Prepare Initial State using ManimAgentState
    initial_state = ManimAgentState(
        initial_user_request=script_segment,
        task_instruction=script_segment,
        context_doc=context_doc,
        rubric=initial_rubric,
        initial_rubric=initial_rubric,
        max_attempts=max_attempts,  # Renamed key
        attempt_number=0,  # Renamed key, start at 0
        general_context=general_context,
        previous_code_attempt=previous_code_attempt,
        enhancement_request=enhancement_request,
        final_command=final_command,
        code=None,
        validation_error=None,
        script_file_path=None,
        execution_success=None,
        failure_summaries=[],
        video_path=None,
        evaluation_result=None,
        evaluation_passed=None,
        error_message=None,
        final_output=None,
        run_output_dir=str(run_output_dir),
        scene_name=scene_name,
        save_generated_code=save_generated_code,
    )
    print("Initial state prepared.")

    # 7. Invoke Graph Asynchronously
    print("\n--- Invoking Manim Agent Graph Asynchronously --- ")
    final_state = None
    try:
        final_state = await app.ainvoke(initial_state)
        result["final_state"] = final_state
        print("\n--- Manim Agent Graph Async Execution Finished ---")

    except Exception as e:
        error_message = f"An unexpected error occurred during async graph execution: {e}"
        print(f"\n--- Manim Agent Graph Async Execution FAILED --- ")
        print(error_message)
        import traceback

        traceback.print_exc()
        result["message"] = error_message
        return result

    # 8. Process Final State
    print("\n--- Processing Final State ---")
    if final_state and final_state.get("evaluation_passed"):
        print("\nSUCCESS: Manim Agent task completed and evaluation passed.")
        result["success"] = True
        result["message"] = "Agent task completed and evaluation passed."

        # --- Adjusted: Use scene_name and run_number for filename ---
        run_output_path_obj = Path(final_state["run_output_dir"])
        run_number_str = run_output_path_obj.name  # Extract run number part (e.g., run_001)

        # Sanitize scene_name for use in filename
        sanitized_scene_name = re.sub(r"\s+", "_", scene_name)  # Replace spaces with underscores
        sanitized_scene_name = re.sub(
            r"[^a-zA-Z0-9_\-]", "", sanitized_scene_name
        )  # Remove non-alphanumeric (allow _ and -)
        sanitized_scene_name = sanitized_scene_name or "scene"  # Use 'scene' if name becomes empty

        base_filename = f"{sanitized_scene_name}"
        # --- End Adjustment ---

        final_output_path_obj = run_output_path_obj / f"{base_filename}.py"
        final_artifact_path_obj = run_output_path_obj / f"{base_filename}.mp4"

        if final_state.get("generated_output"):
            try:
                with open(final_output_path_obj, "w", encoding="utf-8") as f:
                    f.write(final_state["generated_output"])
                final_output_path_str = str(final_output_path_obj)
                print(f"Final output saved to: {final_output_path_str}")
                # Update final state in place for clarity (optional but good practice)
                final_state["final_output_path"] = final_output_path_str
                result["final_output_path"] = final_output_path_str
            except Exception as e:
                print(f"Error saving final output: {e}")
                result["message"] += f" (Warning: Error saving final output: {e})"
        else:
            print("Warning: Evaluation passed, but no final output (code) found.")
            result["message"] += " (Warning: No final output code found)"

        validated_artifact_path = final_state.get("validated_artifact_path")
        if validated_artifact_path:
            # --- Adjusted: Use run_output_dir and handle relative path carefully ---
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
                    result["final_artifact_path"] = final_artifact_path_str
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
            fail_reason = "Unknown (check logs/state)."
            current_attempt = final_state.get("attempt_number", 0)
            max_attempts = final_state.get("max_attempts")
            if not final_state.get("evaluation_passed") and final_state.get("evaluation_result"):
                fail_reason = "Evaluation failed."
            elif final_state.get("validation_error"):
                fail_reason = "Last step resulted in a validation error."
            elif current_attempt >= max_attempts:  # Check if loop ended due to attempts
                fail_reason = f"Maximum attempts ({max_attempts}) reached."  # Updated comment

            result["message"] = f"Agent task failed: {fail_reason}"
        else:
            # This case occurs if app.ainvoke itself failed catastrophically
            # The error message is already set from the exception handler
            pass

    # Log run details
    log_run_details(
        run_output_dir=run_output_dir,
        # Log the final attempt number reached (0 if invoke failed early)
        attempt_number=final_state.get("attempt_number", 0) if final_state else 0,
        node_name="Runner",
        log_category="Run Completion",
        content=f"Agent run finished. Success: {result['success']}. Message: {result['message']}",
    )

    print("\nAsync Manim Agent runner finished.")
    return result
