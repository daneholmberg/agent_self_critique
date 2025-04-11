import re
from typing import Dict, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState
from core.log_utils import log_run_details

MANIM_RULES = """# Manim Cursor Rules

## Do:
- Always encapsulate animations and Mobjects within Scene's `construct()`.
- Use clear, descriptive naming for Scenes and Mobjects (e.g., `IntroScene`, `circle`).
- Recommend modularization (helpers, reusable functions, configs).
- Encourage efficient animations (`Transform`, `ReplacementTransform`) over recreating objects.
- Guide users to official documentation/examples when appropriate.
- Clearly diagnose errors by interpreting stack traces/messages.
- Suggest built-in solutions first (e.g., `Circumscribe`, `FadeIn`).
- Advise breaking complex animations into multiple Scenes.
- Encourage optimization (precomputing heavy calculations, partial rendering).
- Ensure suggested code snippets reflect modern Manim syntax.
- Use classes/functions/methods to organize code.
- put scenes in the scenes folder of the directory you are working in
- Attempt to make the code as efficient as possible for quick rendering without sacrificing quality
- Break long construct() methods into helper functions
- When applicable, use SVGs using SVGMobject, but you'll need to create a temp file to store the SVG and then use that file in the SVGMobject constructor as the first argument
- Use named constants for magic numbers (e.g., COIN_RADIUS = 1.5)
- Optimize 3D meshes with appropriate resolution parameters
- Consider performance: use VGroup instead of many individual mobjects
- Use AnimationGroup and LaggedStart for complex animation sequences
- Use relative positioning instead of absolute positioning when possible
- Don't guess fonts size and such when fitting text and other mobjects to the screen. Calculate everything you need to to make sure everything fits perfectly.
- Take a step back when fixing a bug. Make sure we're not just covering up the bug with a bandaid. Thereby creating more tech debt. Make sure the solution is robust.
- Remember we're using manim CE v0.19.0 (community edition)

## Avoid:
- Using outdated or deprecated methods (`ShowCreation`, old import paths).
- Forgetting to add Mobjects (`self.add`) or animate (`self.play`).
- Mixing old 3b1b-Manim syntax with Manim Community.
- Overcomplicating scene scripts; prefer simple, clear structures.
- Ignoring potential environment/setup issues (e.g., LaTeX, FFmpeg).
- Suggesting overly complex solutions when simpler ones exist.
- Providing incorrect camera operations without appropriate scene classes (e.g., use `ThreeDScene` or `MovingCameraScene`).
- Assuming all constants (colors, directions) are predefined; verify or define explicitly.
- Putting all code in the `construct()` method.
- Don't tell me how to run the animation every single time you change code.
- Overusing Flash animation, consider all of the Indication animations and pick the one that best fits the situation.
- Using Flash animation unless it PERFECTLY fits the situation
- Avoid mixing absolute positioning with relative positioning
- Don't create overly complex animations when simpler ones would work
- Avoid hard-coding coordinates without comments explaining their purpose
- Changing code completely unrelated to the prompt or task given
- Using CYAN, it's not a defined color. If you want to use a undefined color use ManimColor(HEX)
- Use ShowCreation, it's deprecated for Creation"""


class ManimCodeGenerator:
    """Handles the generation of Manim Python code using an LLM."""

    def __init__(self, llm_text_client: ChatGoogleGenerativeAI):
        """
        Initializes the code generator with a text generation LLM client.

        Args:
            llm_text_client: The pre-initialized Langchain LLM client for text generation.
        """
        self.llm_text_client = llm_text_client

    @staticmethod
    def _extract_python_code(text: str) -> Optional[str]:
        """Extracts Python code block from a string (handles ```python ... ```)."""
        # Regex to find code blocks fenced by ```python and ```
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: Maybe the LLM just returned code without fences
        # Basic check: does it look like Python code? (imports, class def)
        if "import" in text and ("class" in text or "def" in text):
            # Be cautious with this fallback, might grab non-code text
            # If the LLM is explicitly prompted for *only* code, this might be okay.
            return text.strip()
        return None

    def _build_generation_prompt(self, state: GraphState, iteration: int) -> str:
        """
        Constructs the prompt for the Manim code generation LLM call.

        Expects 'input_text', 'context_doc', 'general_context', 'final_command',
        'previous_code_attempt' (optional), and 'enhancement_request' (optional)
        in the GraphState. Uses defaults from agent_cfg for general_context/final_command.
        """
        input_text = state["input_text"]  # Segment to animate
        context_doc = state["context_doc"]  # Manim Documentation
        # Check if this is the *start* of an enhancement request
        is_initial_enhancement = state.get("previous_code_attempt") and state.get(
            "enhancement_request"
        )
        # Enhancement request might persist across retries
        enhancement_request = state.get("enhancement_request")

        # Use defaults from config if not present in state
        general_context = state.get("general_context", agent_cfg.DEFAULT_GENERATOR_GENERAL_CONTEXT)
        final_command = state.get("final_command", agent_cfg.DEFAULT_GENERATOR_FINAL_COMMAND)

        # Rubric is not directly used in generation prompt, but passed in state

        prompt_lines = [
            MANIM_RULES,
            f"You are a Manim v0.19 expert. You're goal is to generate the python code for a manim scene.",
            "Follow the instructions in the provided Manim documentation context precisely.",
            "\n--- Manim Documentation Context ---",
            context_doc,
            "--- End Manim Documentation Context ---",
        ]

        # Add General Context if provided (even if it's the default)
        if general_context:  # Check if non-empty after potentially getting default
            prompt_lines.extend(
                [
                    "\n--- General Context ---",
                    general_context.strip(),  # Use strip() for tidiness
                    "--- End General Context ---\n",  # Added newline for spacing
                ]
            )

        # --- Previous Code & Enhancement / Failure Feedback ---
        previous_failed_code = state.get("generated_output")  # Last generated code, may have failed

        if is_initial_enhancement:
            # First iteration of an enhancement request
            previous_code_attempt = state.get(
                "previous_code_attempt"
            )  # The original code to enhance
            prompt_lines.append("--- Previous Code Attempt (For Enhancement) ---")
            prompt_lines.append(f"```python\\n{previous_code_attempt}\\n```")
            prompt_lines.append("--- End Previous Code Attempt ---")
            prompt_lines.append("\\n--- Requested Enhancements ---")
            prompt_lines.append(enhancement_request)
            prompt_lines.append("--- End Requested Enhancements ---")
            prompt_lines.append("\\nPlease enhance the previous code based on the request above.")
        elif iteration > 1:
            # This is a retry (either enhancement or standard)
            # Add previous failed code if it exists
            if previous_failed_code:
                fail_header = "Failed Enhancement" if enhancement_request else "Failed"
                prompt_lines.append(f"\\n--- Previous Code Attempt ({fail_header}) ---")
                prompt_lines.append(f"```python\\n{previous_failed_code}\\n```")
                prompt_lines.append("--- End Previous Code Attempt ---")

            # Add specific feedback based on whether it's an enhancement retry or not
            if enhancement_request:
                prompt_lines.append("--- Enhancement Request (Ongoing) ---")
                prompt_lines.append(enhancement_request)
                prompt_lines.append("--- End Enhancement Request ---")
                self._add_feedback_to_prompt(prompt_lines, state, is_enhancement_retry=True)
            else:
                # Standard retry
                self._add_feedback_to_prompt(prompt_lines, state, is_enhancement_retry=False)

        # --- End Previous Code ---

        # Segment to Animate (Only include if *not* enhancing existing code)
        # An enhancement request implies the context is the previous code + request
        if not enhancement_request:
            prompt_lines.append("\n--- Segment to Animate ---")
            prompt_lines.append(f'"""\n{input_text}\n"""')
            prompt_lines.append("--- End Segment to Animate ---")

        # Final Command from user (or default) - always include
        prompt_lines.append(f"\n--- Task ---")
        prompt_lines.append(final_command.strip())  # Use strip() for tidiness
        prompt_lines.append("--- End Task ---")

        prompt_lines.append(
            "\nGenerate ONLY the complete Python code for the scene, enclosed in ```python ... ``` markers."
        )
        return "\n".join(prompt_lines)

    def _handle_generation_error_and_update_state(
        self,
        updates_to_state: Dict,  # The dict being built and modified
        error_message: str,
        iteration_number: int,
        node_name: str,
        run_output_dir: Path,
        log_event_name: str = "Error",
        error_state_key: str = "validation_error",
    ) -> None:  # Modifies updates_to_state in place
        """Handles common error logging and state updates during code generation."""
        print(f"ERROR: {error_message}")
        log_run_details(
            run_output_dir,
            iteration_number,
            node_name,
            log_event_name,
            error_message,
            is_error=True,
        )
        updates_to_state[error_state_key] = error_message
        # error_history should already be initialized in updates_to_state
        updates_to_state["error_history"].append(f"Iter {iteration_number}: {error_message}")

    def _add_feedback_to_prompt(
        self, prompt_lines: list[str], state: GraphState, is_enhancement_retry: bool
    ):
        """Helper to add validation/evaluation feedback to the prompt lines."""
        feedback_header = (
            "--- Previous Attempt Feedback (Enhancement Retry) ---"
            if is_enhancement_retry
            else "--- Previous Attempt Feedback ---"
        )
        fix_instruction = (
            "Please fix the issues identified above and ensure the enhancement request is met in the new code you generate."
            if is_enhancement_retry
            else "Please fix the issues identified above in the new code you generate."
        )

        prev_validation_error = state.get("validation_error")
        prev_eval_feedback = state.get("evaluation_feedback")

        # Internal helper to format feedback sections
        def _append_feedback_section(header: str, content: Optional[str]):
            if content:
                prompt_lines.append(header)
                prompt_lines.append(f'"""\\n{content}\\n"""')

        if prev_validation_error or prev_eval_feedback:
            prompt_lines.append(f"\\n{feedback_header}")
            # Use the helper
            _append_feedback_section(
                "The previous code failed validation/execution with this error:",
                prev_validation_error,
            )
            _append_feedback_section(
                "The previous code received this evaluation feedback (address these points):",
                prev_eval_feedback,
            )
            prompt_lines.append(fix_instruction)
            prompt_lines.append("--- End Previous Attempt Feedback ---")
        elif state["iteration"] > 0:  # iteration in state is 0-based index before increment
            print(
                f"Warning: Iteration {state['iteration'] + 1} > 1 but no previous error or feedback found in state."
            )

    def generate_manim_code(self, state: GraphState) -> Dict:
        """Generates Manim Python code based on the input text and context using the LLM."""
        node_name = "CodeGenerator"
        run_output_dir = Path(state["run_output_dir"])
        # Iteration number for *this* generation attempt
        current_iteration_number = state["iteration"] + 1

        log_run_details(
            run_output_dir,
            current_iteration_number,
            node_name,
            "Node Entry",
            f"Starting {node_name}...",
        )

        # Make copies of history lists to avoid modifying the original state directly
        error_history = state.get("error_history", [])[:]
        evaluation_history = state.get("evaluation_history", [])[:]

        # Determine if this is the first iteration of an enhancement request
        is_initial_enhancement = state.get("previous_code_attempt") and state.get(
            "enhancement_request"
        )
        persistent_enhancement_request = state.get("enhancement_request")  # Keep if present

        updates_to_state: Dict = {
            "iteration": current_iteration_number,
            "generated_output": None,
            "validation_error": None,
            "validated_artifact_path": None,  # Reset previous artifact path
            "evaluation_feedback": None,  # Reset previous evaluation
            "evaluation_passed": None,  # Reset previous evaluation status
            "error_history": error_history,  # Pass copies
            "evaluation_history": evaluation_history,  # Pass copies
            # --- State Persistence ---
            # Keep enhancement request persistent across retries within the enhancement task
            "enhancement_request": persistent_enhancement_request,
            # Clear previous_code_attempt after the first iteration uses it
            "previous_code_attempt": (
                None if is_initial_enhancement else state.get("previous_code_attempt")
            ),
            # -----------------------
        }

        # --- Construct Prompt using helper ---
        # Ensure required state keys are present (or handled gracefully in _build_generation_prompt)
        required_keys = ["input_text", "context_doc", "run_output_dir"]  # Base requirements
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            error_message = (
                f"Missing required keys in state for code generation: {', '.join(missing_keys)}"
            )
            # Use the helper
            self._handle_generation_error_and_update_state(
                updates_to_state=updates_to_state,
                error_message=error_message,
                iteration_number=current_iteration_number,
                node_name=node_name,
                run_output_dir=run_output_dir,
                log_event_name="State Error",
                error_state_key="infrastructure_error",  # Use specific key for infra issues
            )
            return updates_to_state

        prompt = self._build_generation_prompt(state, current_iteration_number)
        log_run_details(run_output_dir, current_iteration_number, node_name, "LLM Prompt", prompt)

        # --- Call LLM using injected client ---
        llm_output = None
        try:
            print(f"Calling Text Generation LLM: {self.llm_text_client.model}")
            response = self.llm_text_client.invoke(prompt)
            llm_output = response.content
            log_run_details(
                run_output_dir, current_iteration_number, node_name, "LLM Response", llm_output
            )

        except Exception as e:
            error_message = f"Gemini API Error during code generation: {e}"
            # Use the helper
            self._handle_generation_error_and_update_state(
                updates_to_state=updates_to_state,
                error_message=error_message,
                iteration_number=current_iteration_number,
                node_name=node_name,
                run_output_dir=run_output_dir,
                log_event_name="LLM Error",
                error_state_key="validation_error",  # LLM errors treated as validation errors for retry loop
            )
            return updates_to_state

        # --- Parse & Validate Code ---
        log_run_details(
            run_output_dir,
            current_iteration_number,
            node_name,
            "Parsing",
            "Parsing LLM response...",
        )
        extracted_code = self._extract_python_code(llm_output)

        if not extracted_code:
            error_message = (
                "Failed to parse Python code block (```python ... ```) from LLM response."
            )
            # Use the helper
            self._handle_generation_error_and_update_state(
                updates_to_state=updates_to_state,
                error_message=error_message,
                iteration_number=current_iteration_number,
                node_name=node_name,
                run_output_dir=run_output_dir,
                log_event_name="Parsing Error",
            )
            return updates_to_state

        # Set the generated code in the state *before* validation checks
        updates_to_state["generated_output"] = extracted_code

        # Basic validation checks
        if "from manim import" not in extracted_code:
            error_message = "Generated code missing 'from manim import ...' statement."
            # Use the helper
            self._handle_generation_error_and_update_state(
                updates_to_state=updates_to_state,
                error_message=error_message,
                iteration_number=current_iteration_number,
                node_name=node_name,
                run_output_dir=run_output_dir,
                log_event_name="Validation Error",
            )
            return updates_to_state

        expected_class_def = f"class {agent_cfg.GENERATED_SCENE_NAME}(Scene):"
        if not re.search(
            rf"class\s+{agent_cfg.GENERATED_SCENE_NAME}\s*\(.*?Scene.*\):", extracted_code
        ):
            # Allow subclasses like ThreeDScene, MovingCameraScene etc.
            error_message = f"Generated code missing correct Scene class definition inheriting from Scene: expected similar to '{expected_class_def}'."
            # Use the helper
            self._handle_generation_error_and_update_state(
                updates_to_state=updates_to_state,
                error_message=error_message,
                iteration_number=current_iteration_number,
                node_name=node_name,
                run_output_dir=run_output_dir,
                log_event_name="Validation Error",
            )
            return updates_to_state

        # Syntax Check using compile()
        try:
            compile(extracted_code, "<string>", "exec")
            log_run_details(
                run_output_dir,
                current_iteration_number,
                node_name,
                "Syntax Check",
                "Passed basic syntax check.",
            )
        except SyntaxError as e:
            error_message = f"Generated code has SyntaxError: {e}"
            # Use the helper
            self._handle_generation_error_and_update_state(
                updates_to_state=updates_to_state,
                error_message=error_message,
                iteration_number=current_iteration_number,
                node_name=node_name,
                run_output_dir=run_output_dir,
                log_event_name="Syntax Error",
            )
            return updates_to_state

        # --- Update State ---
        log_run_details(
            run_output_dir,
            current_iteration_number,
            node_name,
            "Node Completion",
            "Code generation and basic validation successful.",
        )
        updates_to_state["validation_error"] = None  # Clear validation error on success

        return updates_to_state
