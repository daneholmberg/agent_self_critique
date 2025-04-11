# Manim Automator Agent

## Overview

This project provides an AI-powered agent designed to automate the creation of Manim animations based on script segments. It leverages LangGraph to manage a workflow that includes:

1.  Generating Manim CE Python code using Google's Gemini Pro API based on a text prompt, timestamps, and context documents.
2.  Executing the generated Manim code to render a video.
3.  Evaluating the generated video against a custom rubric using the Gemini Pro Multimodal API.
4.  Iterating the process based on execution errors or evaluation feedback until a satisfactory animation is produced or a maximum iteration limit is reached.

The goal is to significantly speed up the Manim animation workflow for content creators by automating the initial generation, execution, and quality assessment loop.

## Features

*   **AI Code Generation:** Uses Gemini to write Manim Python code from natural language descriptions and timing info.
*   **Automated Execution:** Runs Manim via `subprocess` to render the generated code.
*   **AI Video Evaluation:** Uses Gemini's multimodal capabilities to assess the output video against a user-defined rubric.
*   **Iterative Refinement:** Feeds back execution errors and evaluation results to the AI for improved code generation in subsequent attempts.
*   **State Management:** Uses LangGraph to manage the complex state transitions and loops involved in the process.
*   **Configurable:** Settings like API keys, file paths, quality flags, and iteration limits are managed via configuration files.

## Prerequisites

### Software & Environment

*   **Operating System:** Developed and tested on **WSL2 (Windows Subsystem for Linux)** using an Ubuntu distribution. This is **strongly recommended** due to easier installation of Linux-based dependencies like LaTeX. Other Linux distributions or macOS might work with adjustments, but Windows native is not directly supported due to Manim's dependencies.
*   **Python:** Version 3.9 or higher (check Manim's current requirements).
*   **pip & venv:** Standard Python package management tools.
*   **LaTeX Distribution:** Required by Manim for rendering mathematical text (`Tex`, `MathTex`, etc.). **TeX Live (full)** is highly recommended to avoid missing package issues. Be aware that `texlive-full` is a very large download (several GB) and takes significant time to install.
*   **Optional Manim Build Dependencies:** Libraries like `libcairo2-dev`, `libpango1.0-dev`, `pkg-config` might be needed if `pip install manim` encounters build errors (especially related to cairo/pango).

### Accounts & Services

*   **Google AI API Key:** You need an API key for the Gemini API (including Gemini Pro 1.5 or whichever model supports video input). Obtain this from [Google AI Studio](https://aistudio.google.com/).

## Installation

**IMPORTANT:** Install system dependencies (FFmpeg, LaTeX) *before* installing Python packages, especially Manim.

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd manim_automator
    ```
    (Or simply create your project directory `manim_automator` and place the code files inside).

2.  **Install System Dependencies (WSL2 - Ubuntu/Debian):**
    Open your WSL2 terminal.
    ```bash
    # Update package lists
    sudo apt update

    # Install FFmpeg
    sudo apt install ffmpeg -y

    # Install TeX Live (Full - Recommended but LARGE)
    sudo apt install texlive-full -y
    # --- OR ---
    # Install a smaller subset (May require installing more packages later if errors occur)
    # sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science dvisvgm -y

    # Optional: Install only if pip install manim fails with cairo/pango errors
    # sudo apt install libcairo2-dev libpango1.0-dev pkg-config python3-dev -y

    # Verify installations (should output paths)
    which ffmpeg
    which latex
    which dvisvgm
    ```
    *Note: `texlive-full` installation can take a long time.*

3.  **Set up Python Virtual Environment:**
    Navigate to your project directory (`manim_automator`).
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Python Requirements:**
    Ensure your virtual environment is active.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Verify Python Package Installation:**
    ```bash
    pip show manim langchain langgraph langchain-google-genai
    python -m manim --version
    ```

## Configuration

1.  **API Key (`.env` file):**
    *   Create a file named `.env` in the project root directory (`manim_automator/.env`).
    *   Add your Google AI API key to this file:
        ```dotenv
        GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        ```
    *   **IMPORTANT:** Add `.env` to your `.gitignore` file if using Git to avoid accidentally committing your secret key.

2.  **Core Settings (`config.py`):**
    *   Review the `config.py` file. Most settings have sensible defaults, but you may need to adjust:
        *   `MANIM_VIDEO_OUTPUT_DIR`: **Crucial.** This path (relative to the project root) is where the script *expects* Manim to place its output video files. This often requires adjustment based on your specific Manim version/setup and how it behaves when called via `subprocess` with `--output_file .`. Test by running a simple Manim scene manually first if needed.
        *   `EXPECTED_VIDEO_FILENAME`: The standard filename Manim generates (defaults to `GeneratedScene.mp4`).
        *   `GEMINI_MODEL_NAME`: Ensure this model supports both text generation and video input (e.g., `gemini-1.5-pro-latest`). Check Google's documentation.
        *   `MAX_ITERATIONS`: Controls how many times the agent will try before giving up.
        *   `FINAL_OUTPUT_DIR`: Where successful animations and code are stored.
        *   `RUBRIC_FILE_PATH`: Path to your evaluation rubric.
        *   `MANIM_DOCS_CONTEXT_FILE`: Path to your custom Manim context/style guide.
        *   `VIDEO_EVAL_MAX_SIZE_MB`: Max video size for Gemini evaluation (keep below API limits).

3.  **Evaluation Rubric (`evaluation_rubric.txt`):**
    *   Edit `evaluation_rubric.txt` (or the file specified in `config.py`).
    *   Define clear criteria for Gemini to evaluate the animation videos. Be specific about timing, visual clarity, style adherence, absence of glitches, etc.
    *   **Crucially, instruct Gemini in the `evaluate_video` node's prompt (within `graph_nodes.py`) how to signal overall success (e.g., ending with "Overall Assessment: PASS"). The `decide_after_evaluation` function relies on parsing this signal.**

4.  **Manim Context (`manim3.md` or similar):**
    *   Ensure the file specified in `config.MANIM_DOCS_CONTEXT_FILE` exists and contains your abbreviated Manim documentation or style guide notes. This provides context to the code generation prompt.

## Usage

1.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Set Inputs (Currently Hardcoded):**
    *   Open `run_animation_agent.py`.
    *   Modify the `input_script_segment` and `input_timestamps` variables with the content you want to animate.
    *   *(Future Enhancement: Implement `argparse` to pass these as command-line arguments).*

3.  **Run the Agent:**
    ```bash
    python run_animation_agent.py
    ```

4.  **Monitor Output:**
    *   The script will print status messages as it progresses through the LangGraph nodes (Generate -> Execute -> Evaluate).
    *   If successful, it will print the paths to the final code and video files saved in `config.FINAL_OUTPUT_DIR`.
    *   If it fails after `MAX_ITERATIONS`, it will print error messages, execution history, and evaluation feedback history to help diagnose the issue. Check the console output carefully.
    *   Temporary scripts are saved in `config.MANIM_SCRIPT_DIR`.
    *   Intermediate Manim output (if successful execution occurred) might appear in `config.MANIM_VIDEO_OUTPUT_DIR` or related subdirectories created by Manim, depending on your configuration.