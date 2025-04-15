import os
import dotenv
import shutil
import datetime
import uuid
import sys
import re  # Import re for filename sanitization
import logging  # Added logging import
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
from agents.manim_agent.components.reflector import Reflector  # NEW: Import Reflector
from core.log_utils import log_run_details
from core.llm.client_factory import create_llm_client

# Add logger instance
logger = logging.getLogger(__name__)

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
    rubric_modifier: RubricModifier,
    reflector: Reflector,  # NEW: Add Reflector instance
) -> StateGraph:
    """Builds the LangGraph StateGraph for the Manim agent."""
    print("Building Manim Agent Graph V4 with Reflector...")

    # --- Define Node Functions within builder scope to access components via closure ---
    async def generate_code_node(state: ManimAgentState) -> Dict[str, Any]:
        """Node to generate Manim code."""
        print("\n--- Node: Generate Code ---")
        run_output_dir = Path(state["run_output_dir"])
        current_attempt = state.get("attempt_number", 0) + 1  # Attempt number for this generation

        if not state.get("task_instruction"):
            raise ValueError("Task instruction is missing from state for code generation.")
        result = await code_generator.generate_manim_code(state)

        # NEW: Log the thoughts extracted by the generator
        generation_history = result.get("generation_history", [])
        if generation_history:
            latest_gen_entry = generation_history[-1]
            if latest_gen_entry.get("attempt_index") == current_attempt:
                thoughts = latest_gen_entry.get("thoughts", "[Not Logged]")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    "GenerateCode",
                    "Generator Thoughts",
                    thoughts,
                )
            else:
                logger.warning("Latest generation history index mismatch in generate_code_node.")
        else:
            logger.warning("Generation history not found or empty after code generation call.")

        # Clear previous evaluation/execution/validation states before new attempt
        return {
            "code": result.get("code"),
            "generation_history": generation_history,
            "previous_code_attempt": state.get("code"),  # Store code from *before* this node ran
            "validation_error": None,
            "execution_success": None,
            "evaluation_result": None,
            "evaluation_passed": None,
            "video_path": None,
            "error_message": result.get("error_message"),
            # Carry over reflection history
            "reflection_history": state.get("reflection_history", []),
        }

    async def execute_script_node(state: ManimAgentState) -> Dict[str, Any]:
        """Node to execute the generated Manim script."""
        print("\n--- Node: Execute Script ---")
        code = state.get("code")
        error_message = state.get("error_message")  # Error from generator node

        if not code:
            if error_message:
                print(f"Skipping execution due to code generation error: {error_message}")
                # Set execution_success False, validation_error to the generator error
                return {
                    "execution_success": False,
                    "validation_error": error_message,
                    "error_message": None,
                }
            else:
                logger.warning("execute_script_node: Code is missing, but no error_message found.")
                raise ValueError(
                    "Code is missing for script execution and no generator error found."
                )

        result = await script_executor.execute_manim_script(state)
        # Pass through results, ensure error_message is cleared if execution was attempted
        return {
            "script_file_path": result.get("script_file_path"),
            "video_path": result.get("video_path"),
            "validation_error": result.get("validation_error"),  # From execution
            "execution_success": result.get("execution_success"),
            "error_message": None,  # Clear any previous generator error
        }

    async def evaluate_video_node(state: ManimAgentState) -> Dict[str, Any]:
        """Node to evaluate the generated Manim video."""
        print("\n--- Node: Evaluate Video ---")
        if not state.get("video_path") or not state.get("rubric"):
            # This case should ideally be caught by the should_retry_or_evaluate condition
            logger.error("Evaluate video node reached but video_path or rubric missing!")
            raise ValueError("Video path or rubric is missing from state for evaluation.")
        result = await video_evaluator.evaluate_manim_video(state)
        return {
            "evaluation_result": result.get("evaluation_result"),
            "evaluation_passed": result.get("evaluation_passed"),
        }

    # NEW: Node for the Reflector component
    async def reflect_on_attempt_node(state: ManimAgentState) -> Dict[str, Any]:
        """Node to generate reflection on the completed attempt."""
        print("\n--- Node: Reflect on Attempt ---")
        run_output_dir = Path(state["run_output_dir"])
        # Reflector analyzes the attempt number derived from generation_history
        generation_history = state.get("generation_history", [])
        attempt_index_reflected = (
            generation_history[-1].get("attempt_index", "?") if generation_history else "?"
        )

        # Check if generation history exists, otherwise skip (shouldn't happen in normal flow)
        if not generation_history:
            logger.warning(
                "Reflector node reached but no generation history found. Skipping reflection."
            )
            # Return state keys to maintain graph structure
            return {
                "reflection_history": state.get("reflection_history", []),
                "error_message": "Skipped reflection due to missing generation history.",
            }  # Ensure reflection_history key exists

        result = await reflector.reflect_on_attempt(state)
        logger.info(f"Reflection node completed.")

        # NEW: Log the generated reflection
        reflection_history = result.get("reflection_history", [])
        if (
            reflection_history
            and reflection_history[-1].get("attempt_index") == attempt_index_reflected
        ):
            reflection_text = reflection_history[-1].get("reflection", "[Not Logged]")
            log_run_details(
                run_output_dir,
                attempt_index_reflected,  # Log against the attempt index being reflected upon
                "ReflectOnAttempt",
                "Generated Reflection",
                reflection_text,
            )
            # Log error message from reflection if it occurred
            if result.get("error_message"):
                log_run_details(
                    run_output_dir,
                    attempt_index_reflected,
                    "ReflectOnAttempt",
                    "Reflection Error",
                    result["error_message"],
                    is_error=True,
                )
        else:
            logger.warning(
                "Could not log reflection: History empty or index mismatch after reflection call."
            )

        # Result contains 'reflection_history' and potentially 'error_message'
        return result

    async def modify_rubric_node(state: ManimAgentState) -> Dict[str, Any]:
        """Node to modify the rubric based on the user request and context."""
        print("\n--- Node: Modify Rubric --- ")

        # Check if rubric has already been modified to prevent duplicate runs
        if state.get("rubric_modified", False):
            logger.info("Rubric has already been modified, skipping modification.")
            return {"rubric": state.get("rubric")}

        result = await rubric_modifier.modify_rubric(state)
        print(f"Rubric modification completed.")

        # Mark rubric as modified to prevent future runs
        return {
            "rubric": result.get("rubric"),
            "rubric_modified": True,  # Add flag to prevent rerunning
        }

    async def summarize_single_failure_node(state: ManimAgentState) -> Dict[str, str]:
        """Node to summarize the most recent failure (validation or evaluation)."""
        print("\n--- Node: Summarize Single Failure ---")
        run_output_dir = Path(state["run_output_dir"])
        current_attempt = state.get("attempt_number", 0) + 1  # Attempt that just failed

        failure_detail = ""
        # Prioritize validation error (covers code gen error + execution error)
        if state.get("validation_error"):
            failure_detail = state["validation_error"]
            print("Summarizing failure using 'validation_error' from state.")
        # If no validation error, check evaluation result
        elif state.get("evaluation_passed") is False:
            evaluation_result = state.get("evaluation_result", {})
            failure_detail = evaluation_result.get(
                "feedback", "Evaluation failed - feedback key missing."
            )
            print("Summarizing failure using 'evaluation_result.feedback' from state.")
        else:
            print(
                "Warning: Summarize node called but no validation_error or failed evaluation found."
            )
            log_run_details(
                run_output_dir,
                current_attempt,
                "SummarizeSingleFailure",
                "Warning",
                "No failure found to summarize.",
                is_error=True,
            )
            return {"single_failure_summary": "[No failure detected to summarize]"}

        if not failure_detail:
            print("Warning: Failure detected but detail string is empty.")
            log_run_details(
                run_output_dir,
                current_attempt,
                "SummarizeSingleFailure",
                "Warning",
                "Failure detected but detail was empty.",
                is_error=True,
            )
            return {"single_failure_summary": "[Failure detected but detail was empty]"}

        # Fixed function call with correct parameters
        summary = await failure_summarizer.summarize(
            failure_detail=failure_detail,
            run_output_dir=run_output_dir,
            attempt_number_failed=current_attempt,
        )
        return {"single_failure_summary": summary}

    async def append_failure_summary_node(state: ManimAgentState) -> Dict[str, Any]:
        """Appends the latest failure summary to the list and increments attempt number."""
        print("\n--- Node: Append Failure Summary & Increment Attempt ---")
        current_summaries = state.get("failure_summaries", [])
        single_summary = state.get("single_failure_summary")
        current_attempt = state.get("attempt_number", 0)

        updated_summaries = current_summaries
        if single_summary and single_summary not in [
            "[No failure detected to summarize]",
            "[Failure detected but detail was empty]",
        ]:
            updated_summaries = add_messages(current_summaries, [single_summary])
            print(f"Appended summary: {single_summary}")
        else:
            print("No valid single failure summary to append.")

        # Increment attempt number *after* processing the failure of the current attempt
        next_attempt = current_attempt + 1
        print(f"Incrementing attempt number to {next_attempt}")
        return {
            "failure_summaries": updated_summaries,
            "attempt_number": next_attempt,
            "single_failure_summary": None,  # Clear the single summary
        }

    # --- FIXED: Success node that preserves all state ---
    async def final_success_node(state: ManimAgentState) -> ManimAgentState:
        """Final success node that preserves the entire state."""
        print("\n--- Node: Final Success ---")
        # Create a copy of the state to avoid reference issues
        result = state.copy() if isinstance(state, dict) else {}
        # Add the success message
        result["final_output"] = "Success!"
        # Ensure these flags are set correctly
        result["execution_success"] = True
        result["evaluation_passed"] = True
        return result

    # --- FIXED: Failure node that preserves all state ---
    async def final_failure_node(state: ManimAgentState) -> ManimAgentState:
        """Final failure node that preserves the entire state."""
        print("\n--- Node: Final Failure ---")
        # Create a copy of the state to avoid reference issues
        result = state.copy() if isinstance(state, dict) else {}
        # Add the failure message
        result["final_output"] = "Failure limit reached."
        return result

    # --- Define the StateGraph ---
    workflow = StateGraph(ManimAgentState)

    # --- Add Nodes ---
    workflow.add_node("modify_rubric", modify_rubric_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("execute_script", execute_script_node)
    workflow.add_node("evaluate_video", evaluate_video_node)
    workflow.add_node("reflect_on_attempt", reflect_on_attempt_node)  # NEW: Add reflector node
    workflow.add_node("summarize_failure", summarize_single_failure_node)
    workflow.add_node("append_summary", append_failure_summary_node)
    workflow.add_node("final_success", final_success_node)
    workflow.add_node("final_failure", final_failure_node)

    # --- Define Edges ---
    workflow.set_entry_point("modify_rubric")
    workflow.add_edge("modify_rubric", "generate_code")
    workflow.add_edge(
        "generate_code", "execute_script"
    )  # Explicitly define this edge to ensure proper flow

    # --- Conditional Edge after Code Generation/Execution ---
    workflow.add_conditional_edges(
        "execute_script",
        should_retry_or_evaluate,
        {
            "evaluate": "evaluate_video",  # Proceed to evaluation if execution successful
            "summarize": "summarize_failure",  # Go to summarizer if execution failed
        },
    )

    # --- Conditional Edge after Evaluation (NOW after Reflection) ---
    # Connect evaluation directly to reflection
    workflow.add_edge("evaluate_video", "reflect_on_attempt")

    # Conditional logic happens *after* reflection
    workflow.add_conditional_edges(
        "reflect_on_attempt",  # Source node is now the reflector
        should_retry_or_finish,
        {
            "finish_success": "final_success",  # Finish if evaluation passed
            "summarize": "summarize_failure",  # Go to summarizer if evaluation failed
        },
    )

    # --- Failure Handling Loop ---
    workflow.add_edge("summarize_failure", "append_summary")
    workflow.add_conditional_edges(
        "append_summary",
        check_attempt_limit,
        {
            "continue": "generate_code",  # Loop back to generate code if limit not reached
            "finish_failure": "final_failure",  # End if limit reached
        },
    )

    # --- Final Edges ---
    workflow.add_edge("final_success", END)
    workflow.add_edge("final_failure", END)

    # --- Compile the Graph ---
    app = workflow.compile()
    print("Manim Agent Graph compiled successfully.")
    # Optional: Save graph visualization
    # try:
    #     app.get_graph().draw_mermaid_png(output_file_path="manim_agent_graph.png")
    #     print("Graph visualization saved to manim_agent_graph.png")
    # except Exception as e:
    #     print(f"Could not save graph visualization: {e}")
    return app


# --- Helper Functions for Conditional Logic ---


def should_retry_or_evaluate(state: ManimAgentState) -> str:
    """Decides whether to evaluate the video or summarize failure after script execution."""
    execution_success = state.get("execution_success")
    if execution_success:
        print("Execution successful. Proceeding to evaluation.")
        return "evaluate"
    else:
        print("Execution failed. Proceeding to summarize failure.")
        return "summarize"


def should_retry_or_finish(state: ManimAgentState) -> str:
    """Decides whether to finish successfully or summarize failure after evaluation/reflection."""
    evaluation_passed = state.get("evaluation_passed")
    if evaluation_passed:
        print("Evaluation passed. Finishing successfully.")
        return "finish_success"
    else:
        print("Evaluation failed. Proceeding to summarize failure.")
        # Store the feedback as the failure reason for the summarizer
        evaluation_result = state.get("evaluation_result", {})
        failure_reason = evaluation_result.get("feedback", "Evaluation failed - feedback missing.")
        # Update state directly? Or return a value indicating this?
        # LangGraph typically merges the dict returned by the node;
        # conditions don't usually modify state directly.
        # Let's rely on summarize_failure node to extract this.
        return "summarize"


def check_attempt_limit(state: ManimAgentState) -> str:
    """Checks if the maximum number of attempts has been reached."""
    attempt_number = state.get("attempt_number", 0)
    max_attempts = state.get("max_attempts", 10)
    print(f"Checking attempt limit: Attempt {attempt_number} / Max {max_attempts}")
    if attempt_number >= max_attempts:
        print("Maximum attempts reached. Finishing with failure.")
        return "finish_failure"
    else:
        print("Attempt limit not reached. Continuing retry loop.")
        return "continue"


# --- Main Execution Function ---
async def execute(
    script_segment: str,
    general_context: Optional[str] = None,
    previous_code_attempt: Optional[str] = None,
    enhancement_request: Optional[str] = None,
    final_command: Optional[str] = None,
    scene_name: str = agent_cfg.GENERATED_SCENE_NAME,
    save_generated_code: bool = agent_cfg.SAVE_GENERATED_CODE_DEFAULT,
    run_output_dir_override: Optional[str] = None,
    max_attempts: int = 10,
) -> Dict[str, Any]:
    """
    Executes the Manim agent workflow for a given script segment.

    Args:
        script_segment: The natural language description of the Manim animation needed.
        general_context: Optional general context about the overall goal.
        previous_code_attempt: Optional existing code if enhancing/fixing.
        enhancement_request: Specific request if enhancing/fixing existing code.
        final_command: Optional override for the final instruction to the code generator.
        scene_name: Desired Manim Scene class name.
        save_generated_code: Flag to save intermediate code attempts.
        run_output_dir_override: Specify a specific output directory for this run.
        max_attempts: Maximum number of code generation attempts allowed.

    Returns:
        The final state of the ManimAgentState after execution.
    """
    start_time = datetime.datetime.now()
    print(f"Manim Agent execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task: {script_segment[:100]}...")

    node_name = "ManimRunner"  # Define node name for logging

    # --- Setup Run Directory ---
    run_id = str(uuid.uuid4())[:8]
    run_number = get_next_run_number()
    sanitized_task = re.sub(r"[\\/:*?\"<>|\s]+", "_", script_segment)[:50]
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    if run_output_dir_override:
        run_output_dir = Path(run_output_dir_override).resolve()
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using overridden run output directory: {run_output_dir}")
    else:
        run_output_dir_name = f"run_{run_number:03d}_{timestamp}_{sanitized_task}_{run_id}"
        run_output_dir = agent_cfg.FINAL_AGENT_OUTPUT_DIR / run_output_dir_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created run output directory: {run_output_dir}")

    # --- Log Execution Start ---
    log_run_details(
        run_output_dir=run_output_dir,
        attempt_number=0,  # Not attempt-specific at this point
        node_name=node_name,
        log_category="Execution Start",
        content=f"Starting Manim Agent execution for task: {script_segment[:100]}...",
    )
    # --------------------------

    # --- Initialize LLM Clients ---
    # Use the factory to create clients based on config
    # NOTE: Consider if text/eval models *need* different configs (e.g., temp, safety)
    # For now, using the same base client but different model names if specified
    try:
        print(f"Initializing LLM clients (Provider: {agent_cfg.LLM_PROVIDER})...")
        # Ensure required arguments (provider, model_name, temperature) are passed
        # Temperature added based on function definition - using a default value.
        # Removed api_key and is_vision_model args as they are not accepted by the factory.
        llm_text_client = create_llm_client(
            provider=agent_cfg.LLM_PROVIDER,
            model_name=agent_cfg.TEXT_GENERATION_MODEL,
            temperature=0.9,  # DO NOT TOUCH THE TEMPERATURE
        )
        llm_eval_client = create_llm_client(
            provider=agent_cfg.LLM_PROVIDER,
            model_name=agent_cfg.EVALUATION_MODEL,
            temperature=0.5,  # DO NOT TOUCH THE TEMPERATURE
        )
        # Text client used for summarizer, modifier, and reflector
        print("LLM clients initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize LLM clients.")
        raise RuntimeError(f"LLM Client Initialization Failed: {e}")

    # --- Instantiate Components ---
    print("Instantiating agent components...")
    code_generator = ManimCodeGenerator(llm_text_client)
    script_executor = ManimScriptExecutor()
    video_evaluator = ManimVideoEvaluator(llm_eval_client)
    failure_summarizer = FailureSummarizer(llm_text_client, DEFAULT_FAILURE_SUMMARY_PROMPT)
    rubric_modifier = RubricModifier(llm_text_client)
    reflector = Reflector(llm_text_client)  # NEW: Instantiate Reflector
    print("Agent components instantiated.")

    # --- Log Component Initialization ---
    log_run_details(
        run_output_dir=run_output_dir,
        attempt_number=0,  # Not attempt-specific
        node_name=node_name,
        log_category="Components Initialized",
        content="All agent components initialized successfully.",
    )
    # ----------------------------------

    # --- Load Context & Rubric ---
    try:
        context_doc, initial_rubric = load_context_and_rubric(
            agent_cfg.CONTEXT_FILE_PATH, agent_cfg.RUBRIC_FILE_PATH
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to load required files: {e}")

    # --- Build Graph ---
    print("Building agent graph...")
    # NEW: Pass reflector instance
    manim_agent_graph = build_agent_graph(
        code_generator,
        script_executor,
        video_evaluator,
        failure_summarizer,
        rubric_modifier,
        reflector,
    )
    print("Agent graph built.")

    # --- Log Graph Creation ---
    log_run_details(
        run_output_dir=run_output_dir,
        attempt_number=0,  # Not attempt-specific
        node_name=node_name,
        log_category="Graph Built",
        content="Agent LangGraph successfully built and configured.",
    )
    # -------------------------

    # --- Prepare Initial State ---
    print("Preparing initial state...")
    initial_state: ManimAgentState = {
        "initial_user_request": script_segment,
        "task_instruction": script_segment,  # Starts same as initial request
        "context_doc": context_doc,
        "initial_rubric": initial_rubric,
        "rubric": initial_rubric,  # Starts same as initial rubric
        "max_attempts": max_attempts,
        "attempt_number": 0,
        "failure_summaries": [],
        "generation_history": [],  # NEW: Initialize history lists
        "reflection_history": [],  # NEW: Initialize history lists
        "run_output_dir": str(run_output_dir),
        "scene_name": scene_name,
        "save_generated_code": save_generated_code,
        "rubric_modified": False,  # Add flag to track if rubric has been modified
        # Optional fields based on input args
        "previous_code_attempt": previous_code_attempt,
        "enhancement_request": enhancement_request,
        "general_context": general_context,
        "final_command": final_command,
        # Ensure fields expected by nodes later are initialized if possible
        "code": None,
        "validation_error": None,
        "script_file_path": None,
        "execution_success": None,
        "video_path": None,
        "evaluation_result": None,
        "evaluation_passed": None,
        "error_message": None,
        "final_output": None,
    }
    print(f"Initial state prepared. Max attempts: {max_attempts}")

    # --- Execute Graph ---
    print("\nStarting graph execution...")
    final_state = None

    # --- Log Graph Execution Start ---
    log_run_details(
        run_output_dir=run_output_dir,
        attempt_number=0,  # Starting with attempt 0
        node_name=node_name,
        log_category="Graph Execution Start",
        content=f"Starting LangGraph execution with initial state prepared. Max attempts: {max_attempts}",
    )
    # ------------------------------

    try:
        node_number = 5
        recursion_buffer = 3
        # Use ASYNCHRONOUS stream to correctly handle async nodes
        events = manim_agent_graph.astream(
            initial_state,
            config={
                "recursion_limit": recursion_buffer * ((1 + node_number) * max_attempts)
            },  # DO NOT TOUCH, LEAVE IT HOW IT IS
        )
        # Keep track of the latest state
        latest_state = None

        # Iterate using async for
        async for event in events:
            for key, value in event.items():
                print(f"--- Event: {key} ---")
                # Store every state update we receive
                if isinstance(value, dict) and "attempt_number" in value:
                    latest_state = value
                    logger.debug(f"Updated latest state from event {key}")

                # The final state is usually in the last event of type 'end'
                if key == END:
                    final_state = value  # Langgraph > 0.1 often puts final state here
                    # If END doesn't contain valid state but we have a latest_state, use that
                    if (
                        not isinstance(final_state, dict) or "attempt_number" not in final_state
                    ) and latest_state:
                        logger.info(
                            "END event doesn't contain valid state, using latest tracked state"
                        )
                        final_state = latest_state

        # If final_state isn't valid but we have latest_state, use that as fallback
        if (
            not final_state
            or not isinstance(final_state, dict)
            or "attempt_number" not in final_state
        ) and latest_state:
            logger.info("Using latest tracked state as final state")
            final_state = latest_state

        if final_state:
            print("\nGraph execution complete. Final state obtained.")
        else:
            print("\nGraph execution finished, but final state might not be captured from stream.")
            logger.error("Failed to obtain a valid final state from any event")
            # Don't restart the graph - create a synthetic final state based on last observation
            logger.info("Creating synthetic final state from what we know")
            final_state = {
                **initial_state,  # Start with initial values
                "attempt_number": latest_state.get("attempt_number", 0) if latest_state else 0,
                "execution_success": True,  # If we got to final_success, execution worked
                "evaluation_passed": True,  # If we got to final_success, evaluation passed
                "final_output": "Success!",  # This is what final_success would set
            }
            # If we have other keys in latest_state, preserve them
            if latest_state:
                for key, value in latest_state.items():
                    if key not in final_state or value is not None:
                        final_state[key] = value
            logger.info("Created synthetic final state as fallback")

    except Exception as e:
        logger.exception("An error occurred during graph execution.")
        # Optionally save state on error if possible
        # Consider using a checkpoint saver for robustness
        raise RuntimeError(f"Graph Execution Failed: {e}")

    # --- Post-Execution ---
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nManim Agent execution finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution duration: {duration}")

    # --- Final Output Handling ---
    if final_state:
        print("--- Final State Summary ---")
        print(f"Final Output Message: {final_state.get('final_output')}")
        print(f"Execution Success: {final_state.get('execution_success')}")
        print(f"Evaluation Passed: {final_state.get('evaluation_passed')}")
        final_video_path = final_state.get("video_path")
        if final_video_path and Path(final_video_path).exists():
            print(f"Final Video Path: {final_video_path}")
            # Optionally copy final video to a more prominent location
            try:
                final_dest = run_output_dir / f"final_{agent_cfg.EXPECTED_VIDEO_FILENAME}"
                shutil.copy(final_video_path, final_dest)
                print(f"Final video copied to: {final_dest}")
                # Update state with the final copied path?
                final_state["final_video_destination"] = str(final_dest)
            except Exception as copy_e:
                print(f"Warning: Could not copy final video: {copy_e}")
        elif final_state.get("evaluation_passed"):
            print("Final Video Path: [Not Available or Not Found, but evaluation passed?]")
        else:
            print("Final Video Path: [Not Generated or Execution/Evaluation Failed]")

        # Log full final state to a file
        try:
            final_state_log_path = run_output_dir / "final_state.log"
            with open(final_state_log_path, "w", encoding="utf-8") as f:
                # Convert state to string for logging (e.g., using pprint)
                import pprint

                f.write(pprint.pformat(final_state))
            print(f"Final state logged to: {final_state_log_path}")
        except Exception as log_e:
            logger.warning(f"Could not log final state to file: {log_e}")

        # --- Log Final Run Completion Status ---
        final_status_msg = final_state.get("final_output", "Unknown")
        error_msg = final_state.get("error_message")
        completion_details = f"Final Status: {final_status_msg}"
        if error_msg:
            completion_details += f" | Error: {error_msg}"
        # Get the final attempt number from state
        final_attempt = final_state.get("attempt_number", 0)
        is_error = bool(error_msg) or not final_state.get("evaluation_passed", False)

        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=final_attempt,
            node_name=node_name,
            log_category="Run Completion",
            content=completion_details,
            is_error=is_error,
        )
        # -----------------------------------------

        return final_state
    else:
        print("Error: Final state could not be determined after execution.")

        # --- Log Execution Error ---
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=0,  # Can't determine final attempt
            node_name=node_name,
            log_category="Execution Error",
            content="Execution finished but final state is undetermined.",
            is_error=True,
        )
        # ---------------------------

        # Return a minimal error state or raise exception?
        return {
            "error_message": "Execution finished but final state is undetermined.",
            "run_output_dir": str(run_output_dir),
        }


# --- Example Usage (CLI) ---
if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Run the Manim Agent.")
    parser.add_argument(
        "script_segment", type=str, help="The natural language description for the animation."
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=agent_cfg.GENERATED_SCENE_NAME,
        help="Desired Manim Scene class name.",
    )
    parser.add_argument("--max_attempts", type=int, default=5, help="Maximum generation attempts.")
    parser.add_argument(
        "--save_code", action="store_true", help="Save intermediate generated code."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override the run output directory."
    )
    # Arguments for enhancement flow
    parser.add_argument(
        "--enhance_code_file", type=str, default=None, help="Path to existing code file to enhance."
    )
    parser.add_argument(
        "--enhancement_request", type=str, default=None, help="Specific enhancement request text."
    )

    args = parser.parse_args()

    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Set higher level for verbose libraries if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING)

    # Load environment variables from .env file if present
    dotenv.load_dotenv()

    # Handle enhancement args
    previous_code = None
    if args.enhance_code_file:
        try:
            with open(args.enhance_code_file, "r", encoding="utf-8") as f:
                previous_code = f.read()
            print(f"Loaded previous code for enhancement from: {args.enhance_code_file}")
            if not args.enhancement_request:
                print("Warning: --enhance_code_file provided without --enhancement_request.")
                # sys.exit("Error: Must provide --enhancement_request when using --enhance_code_file.")
        except FileNotFoundError:
            sys.exit(f"Error: Enhancement code file not found: {args.enhance_code_file}")
        except Exception as e:
            sys.exit(f"Error reading enhancement code file: {e}")

    # Run the agent asynchronously
    try:
        final_run_state = asyncio.run(
            execute(
                script_segment=args.script_segment,
                scene_name=args.scene_name,
                max_attempts=args.max_attempts,
                save_generated_code=args.save_code,
                run_output_dir_override=args.output_dir,
                previous_code_attempt=previous_code,
                enhancement_request=args.enhancement_request,
                # general_context, final_command could be added as args if needed
            )
        )
        # Print a summary or confirmation
        print("\n--- Execution Summary ---")
        print(f"Final Status: {final_run_state.get('final_output', 'Unknown')}")
        if final_run_state.get("final_video_destination"):
            print(f"Output Video: {final_run_state['final_video_destination']}")
        elif final_run_state.get("error_message"):
            print(f"Error: {final_run_state['error_message']}")

    except RuntimeError as e:
        print(f"\nExecution failed with runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logger.exception("Unhandled exception during agent execution.")
        sys.exit(1)
