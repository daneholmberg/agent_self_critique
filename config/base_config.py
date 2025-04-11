import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root if it exists
# Assumes .env is located in the parent directory of this config file's directory
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# --- Project Root ---
# Calculate the project root directory (assuming this file is in agent_automator/config/)
PROJECT_ROOT = Path(__file__).parent.parent
BASE_DIR = PROJECT_ROOT  # Alias for consistency/use as cwd base

# --- Environment Variable Names ---
# Define constants for the *names* of the environment variables we'll look for.
# This makes it easier to manage and potentially override which env var corresponds to which setting.
ENV_GEMINI_API_KEY_NAME = "GEMINI_API_KEY_NAME"  # Env var that HOLDS THE NAME of the key variable
ENV_GEMINI_DEFAULT_MODEL_NAME = "GEMINI_DEFAULT_MODEL_NAME"
ENV_MAX_ITERATIONS = "MAX_ITERATIONS"
ENV_BASE_OUTPUT_DIR = "BASE_OUTPUT_DIR"  # Base directory for all task outputs

# --- Core Configuration Values ---
# Load the actual configuration values from environment variables using the names defined above.
# Provides sensible defaults if the environment variables are not set.

# The *name* of the environment variable that holds the actual API key.
# Defaults to looking for an env var named 'GOOGLE_API_KEY'.
GEMINI_API_KEY_NAME = os.getenv(ENV_GEMINI_API_KEY_NAME, "GOOGLE_API_KEY")

# Default Gemini model to use if a task doesn't specify one.
GEMINI_DEFAULT_MODEL_NAME = os.getenv(ENV_GEMINI_DEFAULT_MODEL_NAME, "gemini-2.5-pro-preview-03-25")

# Default maximum number of iterations for the generation/evaluation loop.
MAX_ITERATIONS = int(os.getenv(ENV_MAX_ITERATIONS, "5"))  # Ensure it's an integer

# Base directory for storing outputs from different tasks (relative to project root).
# Tasks might create subdirectories within this base directory.
BASE_OUTPUT_DIR = PROJECT_ROOT / os.getenv(ENV_BASE_OUTPUT_DIR, "outputs")


# --- Helper Function for API Key Retrieval ---
def get_gemini_api_key() -> str:
    """
    Retrieves the actual Gemini API key from the environment variable specified by GEMINI_API_KEY_NAME.

    Raises:
        ValueError: If the environment variable named by GEMINI_API_KEY_NAME is not set
                    or if the environment variable it points to doesn't contain the key.
                    or if the environment variable it points to is empty.
    """
    # Get the *name* of the env var that supposedly holds the key
    api_key_var_name = os.getenv(
        ENV_GEMINI_API_KEY_NAME, "GOOGLE_API_KEY"
    )  # Defaults to GOOGLE_API_KEY

    if not api_key_var_name:
        # This case is unlikely if the default is set, but good practice
        raise ValueError(
            f"The environment variable '{ENV_GEMINI_API_KEY_NAME}' (which should contain the *name* of the API key variable) is not set and no default is available."
        )

    # Get the *actual key* from the environment variable named by api_key_var_name
    api_key = os.getenv(api_key_var_name)

    if not api_key:
        raise ValueError(
            f"API key environment variable '{api_key_var_name}' (specified by '{ENV_GEMINI_API_KEY_NAME}' or default) is not set or is empty."
        )

    return api_key


# --- Optional: Ensure Base Output Directory Exists ---
# It's generally better practice to create directories in the main application
# logic or task runner, but ensuring the *base* output directory exists here
# can sometimes be convenient. Let's keep it commented out for now to adhere strictly
# to the principle of config files not having side effects.
# os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
