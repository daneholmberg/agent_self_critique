import os
import pathlib
import dotenv
from pathlib import Path
from typing import List, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config.base_config import (
    PROJECT_ROOT,
    BASE_OUTPUT_DIR,
    get_gemini_api_key,
    GEMINI_DEFAULT_MODEL_NAME,
)

# Load environment variables from .env file if it exists
# This is potentially redundant if base_config already loaded it, but ensures it's loaded.
dotenv.load_dotenv()

# Agent specific name
AGENT_NAME = "manim_agent"

# --- Manim Environment Variable Names ---
# These constants define the names of the environment variables used to configure Manim execution.
ENV_OUTPUT_SUBDIR = "MANIM_OUTPUT_SUBDIR"
ENV_TEMP_SCRIPT_SUBDIR = "MANIM_TEMP_SCRIPT_SUBDIR"
ENV_VIDEO_OUTPUT_RELATIVE_PATH = (
    "MANIM_VIDEO_OUTPUT_RELATIVE_PATH"  # Relative to the temp script dir where Manim runs
)
ENV_EXPECTED_VIDEO_FILENAME = "MANIM_EXPECTED_VIDEO_FILENAME"
ENV_QUALITY_FLAG = "MANIM_QUALITY_FLAG"  # Manim quality flag (e.g., -pql, -pqm, -pqh)
ENV_CONTEXT_FILE = "MANIM_CONTEXT_FILE"  # Path to the context file (relative to PROJECT_ROOT)
ENV_RUBRIC_FILE = "MANIM_RUBRIC_FILE"  # Path to the rubric file (relative to PROJECT_ROOT)
ENV_VIDEO_EVAL_MAX_SIZE_MB = "MANIM_VIDEO_EVAL_MAX_SIZE_MB"  # Max video size in MB for evaluation
ENV_GENERATED_SCENE_NAME = "MANIM_GENERATED_SCENE_NAME"  # Default scene name if not specified
ENV_SAVE_GENERATED_CODE = "MANIM_SAVE_GENERATED_CODE"  # Flag to save generated code per iteration
# NEW: Video Evaluation Frame Extraction Config
ENV_VIDEO_EVAL_FPS = "MANIM_VIDEO_EVAL_FPS"  # Frames per second to extract for evaluation
ENV_VIDEO_EVAL_MAX_FRAMES = "MANIM_VIDEO_EVAL_MAX_FRAMES"  # Max frames to send for evaluation
ENV_MANIM_EXECUTION_TIMEOUT_SECONDS = (
    "MANIM_EXECUTION_TIMEOUT_SECONDS"  # Timeout for Manim execution
)
# NEW: LLM Configuration Environment Variable Names
ENV_LLM_PROVIDER = "MANIM_LLM_PROVIDER"
ENV_TEXT_MODEL = "MANIM_TEXT_MODEL"
ENV_EVAL_MODEL = "MANIM_EVAL_MODEL"

# --- Load Actual Manim Configuration Values ---
# Load configuration values from environment variables, falling back to defaults if not set.

# Subdirectory within BASE_OUTPUT_DIR for final Manim animations
OUTPUT_SUBDIR_NAME = os.getenv(ENV_OUTPUT_SUBDIR, "manim_animations")

# Subdirectory within BASE_OUTPUT_DIR for temporary Manim script files
TEMP_SCRIPT_SUBDIR_NAME = os.getenv(ENV_TEMP_SCRIPT_SUBDIR, "temp_manim_scripts")

# Manim's internal output path relative to where the script is executed (TEMP_SCRIPT_DIR).
# This needs careful handling as it depends on Manim's configuration/flags.
# Default assumes standard Manim output structure with -pql quality. Adjust if necessary.
MANIM_VIDEO_OUTPUT_RELATIVE_PATH = os.getenv(
    ENV_VIDEO_OUTPUT_RELATIVE_PATH, "media/videos/GeneratedScene/480p15"
)

# The expected filename of the video generated by Manim within its output directory.
EXPECTED_VIDEO_FILENAME = os.getenv(ENV_EXPECTED_VIDEO_FILENAME, "GeneratedScene.mp4")

# Manim quality flag (e.g., -pql for 480p15, -pqm for 720p30, -pqh for 1080p60)
MANIM_QUALITY_FLAG = os.getenv(ENV_QUALITY_FLAG, "-pql")

# Path to the context file providing background information for generation
CONTEXT_FILE_PATH = PROJECT_ROOT / os.getenv(ENV_CONTEXT_FILE, "context_docs/manim19.md")

# Path to the rubric file used for evaluating the generated Manim animation
RUBRIC_FILE_PATH = PROJECT_ROOT / os.getenv(ENV_RUBRIC_FILE, "context_docs/manim_rubric.txt")

# Maximum allowed size (in Megabytes) for the generated video during evaluation
VIDEO_EVAL_MAX_SIZE_MB = int(os.getenv(ENV_VIDEO_EVAL_MAX_SIZE_MB, "19"))

# The default name for the Manim Scene class to be generated if not provided in the request
GENERATED_SCENE_NAME = os.getenv(ENV_GENERATED_SCENE_NAME, "GeneratedScene")
# NEW: Video Evaluation Frame Extraction Config
VIDEO_EVAL_FRAMES_PER_SECOND = int(os.getenv(ENV_VIDEO_EVAL_FPS, "1"))  # Default to 1 FPS
VIDEO_EVAL_MAX_FRAMES = int(os.getenv(ENV_VIDEO_EVAL_MAX_FRAMES, "40"))  # Default max 20 frames

# Flag to determine if generated code should be saved permanently per iteration
# Defaults to False (0). Set to 1 or true in .env to enable.
SAVE_GENERATED_CODE_DEFAULT = os.getenv(ENV_SAVE_GENERATED_CODE, "0").lower() in (
    "true",
    "1",
    "t",
)

MANIM_EXECUTION_TIMEOUT_SECONDS = int(os.getenv(ENV_MANIM_EXECUTION_TIMEOUT_SECONDS, "120"))

# NEW: Load LLM Configuration
# Provider (currently only 'google' supported by the factory)
LLM_PROVIDER = os.getenv(ENV_LLM_PROVIDER, "google")
# Text Generation Model Configuration
TEXT_GENERATION_MODEL = os.getenv(ENV_TEXT_MODEL, GEMINI_DEFAULT_MODEL_NAME)
# Evaluation Model Configuration (Defaulting to the same as text generation for now)
# Ensure the model used supports vision if needed for video evaluation
EVALUATION_MODEL = os.getenv(ENV_EVAL_MODEL, GEMINI_DEFAULT_MODEL_NAME)


# --- Derived Paths ---
# These paths are constructed based on the loaded configuration values.

# Final destination directory for the successfully generated Manim videos
FINAL_AGENT_OUTPUT_DIR = BASE_OUTPUT_DIR / OUTPUT_SUBDIR_NAME

# Directory where temporary Manim scripts (*.py) will be saved before execution
TEMP_SCRIPT_DIR = BASE_OUTPUT_DIR / TEMP_SCRIPT_SUBDIR_NAME

# --- Ensure Directories Exist ---
# Create the necessary output and temporary directories if they don't already exist.
FINAL_AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# --- Validation (Optional but Recommended) ---
# Add checks here if needed, e.g., ensure context/rubric files exist
if not CONTEXT_FILE_PATH.is_file():
    print(f"Warning: Manim context file not found at {CONTEXT_FILE_PATH}")
# Uncomment the following lines if the rubric file is mandatory
# if not RUBRIC_FILE_PATH.is_file():
#     # Consider raising an error or logging more severely if the rubric is critical
#     print(f"Warning: Manim rubric file not found at {RUBRIC_FILE_PATH}")

# You might want to add more validation here, e.g.:
# - Check if MANIM_QUALITY_FLAG is one of the expected values.
# - Ensure VIDEO_EVAL_MAX_SIZE_MB is a positive integer.

# Constants
MANIM_VERSION_TARGET = "0.19.0"  # Target Manim CE version
MANIM_MODULE_NAME = "manim_scene.py"
DEFAULT_OUTPUT_DIR = Path("outputs/manim_agent")
DEFAULT_LOG_FILE = "manim_agent_run.log"
DEFAULT_MANIM_RENDER_QUALITY = "low"  # Options: low, medium, high, production, 4k

# NEW: Default prompts for Code Generator
DEFAULT_GENERATOR_GENERAL_CONTEXT = """
Consider the overall tone and purpose of the script segment.
Prioritize clarity and visual appeal in the animation.
Keep animations concise and directly relevant to the text.
Avoid overuse of text. Text can be effective as titles or for emphasizing key points as a summary, but should be used sparingly.
Use Manim's capabilities effectively, but avoid overly complex effects unless necessary.
"""

# MODIFIED: Use {scene_name} placeholder
DEFAULT_GENERATOR_FINAL_COMMAND = """
Generate the Python code for a Manim scene that effectively visualizes the provided 'Primary Task Instruction'.
Ensure the code defines a class named '{scene_name}' EXACTLY LETTER FOR LETTER THIS IS CRUCIAL.
Adhere strictly to Manim v0.19 syntax and best practices based on the documentation context.
Produce clean, readable, and functional Python code.
"""

# NEW: Default prompt for Failure Summarizer
DEFAULT_FAILURE_SUMMARY_PROMPT = (
    "Failure Reason:\n{failure_detail}\n\n"
    "Given the failure reason encountered during Manim video generation, "
    "provide a very concise, token-optimized summary. I want it to be as concise as possible while still being informative enough to make"
    "sure we don't repeat the issue later on.\n\n"
    "Do not include apologies or suggestions for fixes, just state the problem clearly.\n\n"
    "Concise Summary:"
)


class ManimAgentState(TypedDict, total=False):
    """State for the Manim agent graph."""

    # --- Core Task Information ---
    initial_user_request: str
    task_instruction: str
    parsed_instruction: Optional[dict] = None  # Parsed structured instruction
    task_decomposition: Optional[List[dict]] = None  # List of sub-tasks if decomposed
    context_doc: str  # Manim documentation context (Required for prompts)

    # --- Code Generation & Execution ---
    code: Optional[str] = None  # Generated Manim code
    previous_code_attempt: Optional[str] = None  # Store the last code attempt
    # --- NEW: History Tracking ---
    # Stores the thoughts and code generated by the code_generator at each attempt.
    generation_history: List[
        dict
    ]  # Expected format: [{"attempt_index": int, "thoughts": str, "code": str}]
    # Stores the reflections generated by the reflector node after each evaluation.
    reflection_history: List[dict]  # Expected format: [{"attempt_index": int, "reflection": str}]
    # --- Execution & Validation ---
    validation_error: Optional[str] = None  # Error message from Manim execution
    script_file_path: Optional[str] = None  # Path to the saved Manim script
    execution_success: Optional[bool] = None  # Flag indicating successful Manim run
    failure_summaries: Annotated[List[str], add_messages]  # Accumulated summaries of failures
    single_failure_summary: Optional[str] = None  # Temporary storage for the latest summary

    # --- Rubric & Evaluation ---
    rubric: Optional[str] = None  # The evaluation rubric
    initial_rubric: Optional[str] = None  # Store the original rubric for reference
    video_path: Optional[str] = None  # Path to the generated video
    evaluation_result: Optional[dict] = (
        None  # Results from the video evaluator (pass/fail, reasoning)
    )
    evaluation_passed: Optional[bool] = None  # Flag indicating successful evaluation

    # --- Control Flow & Meta ---
    # attempt_number: Tracks the number of generation attempts made for the *current* task/request.
    # Starts at 0, incremented *after* a failure (validation, execution, or evaluation) before retry.
    # Used to limit total attempts via max_attempts.
    attempt_number: int = 0
    # max_attempts: The maximum number of attempts allowed before giving up.
    max_attempts: int

    error_message: Optional[str] = None  # General error message storage
    final_output: Optional[str] = None  # Final message or result for the user

    # --- Run Configuration (Passed through) ---
    run_output_dir: str  # Directory for run-specific output
    scene_name: str  # Desired scene name for output files
    save_generated_code: bool  # Flag to save intermediate code

    # --- Enhancement Request (Optional) ---
    enhancement_request: Optional[str] = None
    # --- General Context (Optional) ---
    general_context: Optional[str] = None
    # --- Final Command (Optional) ---
    final_command: Optional[str] = None


# --- Model Configurations ---
# ... existing code ...
