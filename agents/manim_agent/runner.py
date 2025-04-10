import os
import dotenv
import shutil
import datetime
import uuid
import sys
from typing import Tuple, Optional
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


def get_multiline_input(prompt_message: str) -> str:
    """Prompts the user for multi-line input and returns the combined string.
    Reads from stdin until EOF (Ctrl+D/Ctrl+Z+Enter) is detected.
    """
    print(prompt_message)
    print("(Paste your text, then press Ctrl+D (Unix) or Ctrl+Z+Enter (Windows) to finish)")
    try:
        input_text = sys.stdin.read()
    except Exception as e:
        # Catch potential issues, though stdin.read() is generally robust
        print(f"Error reading input: {e}")
        return ""
    return input_text.strip()  # Remove leading/trailing whitespace


def execute(input_metadata: Optional[str] = None):
    """
    Main execution function for the Manim agent.
    Handles interactive input specific to Manim generation.

    Args:
        input_metadata: Optional metadata string passed from main.py.
    """
    agent_name = "manim_agent"
    print(f"Starting Manim Agent execution via runner.")

    # 1. Get Manim-specific Input Interactively
    script_segment_to_animate = get_multiline_input("Enter the specific script segment to animate:")
    if not script_segment_to_animate:
        print("Error: No script segment provided.")
        sys.exit(1)

    # Optionally, ask for full script context (can be empty)
    full_script_context = get_multiline_input(
        "(Optional) Enter the full script context for better understanding:"
    )

    # Combine inputs for the state
    # We'll use the specific segment as the main 'input_text'
    prompt_text = script_segment_to_animate
    # The full_script_context will be passed directly to the generator
    print("Received script segment and context.")

    # 2. Load Base Config & Environment Variables (Should be loaded by main.py ideally, but load here for now)
    #    If main.py handles dotenv loading, this can be removed.
    dotenv.load_dotenv()
    try:
        api_key = base_cfg.get_gemini_api_key()
        if not api_key:
            raise ValueError("Gemini API key is missing.")
        print("Gemini API key loaded successfully.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure the GOOGLE_API_KEY environment variable is set correctly.")
        return  # Exit if API key is missing

    # 2. Instantiate Manim Agent Components
    try:
        print("Instantiating Manim agent components...")
        text_llm, eval_llm = llm_clients.get_llm_clients()

        # Pass the full script context to the code generator here
        code_generator = ManimCodeGenerator(
            llm_text_client=text_llm, script_context=full_script_context  # Pass the context
        )
        script_executor = ManimScriptExecutor()
        video_evaluator = ManimVideoEvaluator(llm_eval_client=eval_llm)

        print("Manim agent components instantiated.")
    except Exception as e:
        print(f"Error during component instantiation: {e}")
        return

    # 3. Get Manim Agent Node Functions (Methods)
    # Directly referencing the known methods for the Manim agent components
    generate_func = code_generator.generate_manim_code
    validate_func = script_executor.execute_manim_script
    evaluate_func = video_evaluator.evaluate_manim_video
    print("Node functions retrieved.")

    # 4. Prepare Paths & Context
    try:
        # Create directories (idempotent)
        os.makedirs(agent_cfg.FINAL_AGENT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(agent_cfg.TEMP_SCRIPT_DIR, exist_ok=True)
        print(f"Ensured output directory exists: {agent_cfg.FINAL_AGENT_OUTPUT_DIR}")
        print(f"Ensured temp directory exists: {agent_cfg.TEMP_SCRIPT_DIR}")

        # Load context and rubric using agent-specific config paths
        context_doc, rubric = load_context_and_rubric(
            agent_cfg.CONTEXT_FILE_PATH, agent_cfg.RUBRIC_FILE_PATH
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error during path/context preparation: {e}")
        return

    # 5. Build Graph
    print("Building the execution graph...")
    app = build_graph(generate_func, validate_func, evaluate_func)
    print("Graph built successfully.")

    # 6. Prepare Initial State (Use provided inputs)
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
    )
    print("Initial state prepared.")

    # 7. Invoke Graph
    print("\n--- Invoking Manim Agent Graph --- ")
    try:
        final_state = app.invoke(initial_state)
        print("\n--- Manim Agent Graph Execution Finished ---")
    except Exception as e:
        print(f"\n--- Manim Agent Graph Execution FAILED --- ")
        print(f"An unexpected error occurred during graph execution: {e}")
        return

    # 8. Process Final State
    print("\n--- Processing Final State ---")
    if final_state and final_state.get("evaluation_passed"):
        print("\nSUCCESS: Manim Agent task completed and evaluation passed.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        base_filename = f"{agent_name}_{timestamp}_{unique_id}"
        final_output_path = agent_cfg.FINAL_AGENT_OUTPUT_DIR / f"{base_filename}_code.py"
        final_artifact_path = agent_cfg.FINAL_AGENT_OUTPUT_DIR / f"{base_filename}_video.mp4"

        if final_state.get("generated_output"):
            try:
                with open(final_output_path, "w", encoding="utf-8") as f:
                    f.write(final_state["generated_output"])
                print(f"Final output saved to: {final_output_path}")
                final_state["final_output_path"] = str(final_output_path)
            except Exception as e:
                print(f"Error saving final output: {e}")
        else:
            print("Warning: Evaluation passed, but no final output (code) found.")

        validated_artifact_path = final_state.get("validated_artifact_path")
        if validated_artifact_path:
            source_artifact_full_path = base_cfg.PROJECT_ROOT / validated_artifact_path
            if source_artifact_full_path.exists():
                try:
                    print(
                        f"Copying artifact from {source_artifact_full_path} to {final_artifact_path}"
                    )
                    shutil.copy2(source_artifact_full_path, final_artifact_path)
                    print(f"Final artifact saved to: {final_artifact_path}")
                    final_state["final_artifact_path"] = str(final_artifact_path)
                    try:
                        os.remove(source_artifact_full_path)
                        print(f"Removed temporary artifact: {source_artifact_full_path}")
                    except OSError as e:
                        print(
                            f"Warning: Could not remove temp artifact {source_artifact_full_path}: {e}"
                        )
                except Exception as e:
                    print(f"Error copying/saving final artifact: {e}")
            else:
                print(f"Warning: Artifact path {source_artifact_full_path} not found.")
        else:
            print("Warning: Evaluation passed, but no validated artifact path found.")
    else:
        print("\nFAILURE: Manim Agent task did not pass evaluation or encountered errors.")
        if final_state:
            # Print failure reasons and histories
            if not final_state.get("evaluation_passed") and final_state.get("evaluation_feedback"):
                print("Reason: Evaluation failed.")
            elif final_state.get("validation_error"):
                print("Reason: Last step resulted in a validation error.")
            else:
                print("Reason: Unknown (check logs/state, possibly max iterations).")
            print("\n--- Error History ---")
            for i, error in enumerate(final_state.get("error_history", [])):
                print(f"{i+1}: {error}\n")
            print("\n--- Evaluation History ---")
            for i, evaluation in enumerate(final_state.get("evaluation_history", [])):
                print(f"{i+1}: {evaluation}\n")
        else:
            print("Reason: Final state is empty or None (early graph error).")

    print("\nManim Agent execution finished.")
