import re
from typing import Dict, Optional, Tuple
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
    def _extract_code_and_thoughts(text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts Python code block (```python ... ```) and thoughts block
        (```thoughts ... ```) from a string.
        """
        # Regex to find code blocks fenced by ```python and ```
        code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        extracted_code = code_match.group(1).strip() if code_match else None

        # Regex to find thoughts blocks fenced by ```thoughts and ```
        thoughts_match = re.search(r"```thoughts\n(.*?)```", text, re.DOTALL)
        extracted_thoughts = thoughts_match.group(1).strip() if thoughts_match else None

        # Fallback for code: Maybe the LLM just returned code without fences
        if not extracted_code and "import" in text and ("class" in text or "def" in text):
            # Try to extract based on known start/end patterns if thoughts exist
            if extracted_thoughts:
                # If thoughts were found, assume code is before or after it
                thoughts_block = thoughts_match.group(0)  # Get the full ```thoughts...``` block
                code_candidate = text.replace(thoughts_block, "").strip()
                # Basic check: does it look like Python code?
                if "import" in code_candidate and (
                    "class" in code_candidate or "def" in code_candidate
                ):
                    extracted_code = code_candidate
            else:
                # If no thoughts, assume the whole text might be code (use with caution)
                extracted_code = text.strip()

        return extracted_code, extracted_thoughts

    def _build_generation_prompt(self, state: GraphState, iteration: int) -> str:
        """
        Constructs the prompt for the Manim code generation LLM call.

        Expects 'input_text', 'context_doc', 'general_context', 'final_command',
        'previous_code_attempt' (optional), 'enhancement_request' (optional),
        and 'thought_history' (optional) in the GraphState.
        Uses defaults from agent_cfg for general_context/final_command.
        """
        input_text = state["input_text"]  # Segment to animate
        context_doc = state["context_doc"]  # Manim Documentation
        thought_history = state.get("thought_history", [])  # Get thought history

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

        # --- Add Thought History --- (If it exists)
        if thought_history:
            prompt_lines.append("\n--- History of Previous Thoughts & Outcomes ---")
            for i, thought_entry in enumerate(thought_history):
                # Add the full thought entry, which already includes iteration and outcome
                prompt_lines.append(thought_entry)
            prompt_lines.append("--- End History of Previous Thoughts & Outcomes ---\n")

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

        prompt_lines.append("\n--- Segment to Animate ---")
        prompt_lines.append(f'"""\n{input_text}\n"""')
        prompt_lines.append("--- End Segment to Animate ---")

        # --- Determine Final Task Instruction ---
        task_content = None
        # Attempt refinement only on retries (iteration > 1 originally, now checks current_iteration_number)
        if iteration > 1 and (state.get("validation_error") or state.get("evaluation_feedback")):
            # Call the helper to get the refined task
            task_content = self._generate_refined_task(state, iteration)  # Pass current iter num

        # Fallback to original command if refinement wasn't attempted (iter 1) or if it failed (returned None)
        if not task_content:
            # Always fall back to the original final command (or its default).
            # The prompt structure already presents enhancement details separately if applicable.
            task_content = final_command  # Use the final_command retrieved earlier

        # --- Append Final Task ---
        prompt_lines.append(f"\\n--- Task ---")
        prompt_lines.append(task_content.strip())  # Use the determined task content
        prompt_lines.append("--- End Task ---")

        prompt_lines.append(
            "\nGenerate ONLY the complete Python code for the scene, enclosed in ```python ... ``` markers."
            " Additionally, provide your thought process, reflections on previous attempts (if any), and plan for this generation attempt in a separate ```thoughts ... ``` block. (note this is different from your internal thinking before the answer)"
        )
        return "\n".join(prompt_lines)

    def _generate_refined_task(self, state: GraphState, iteration: int) -> Optional[str]:
        """
        Uses the LLM to generate a refined, specific task instruction based on
        previous errors or feedback.

        Args:
            state: The current graph state containing feedback.
            iteration: The current iteration number (1-based).

        Returns:
            The refined task string, or None if no feedback exists or an error occurs.
        """
        validation_error = state.get("validation_error")
        evaluation_feedback = state.get("evaluation_feedback")
        run_output_dir = Path(state["run_output_dir"])
        node_name = "CodeGenerator_TaskRefiner"  # Distinguish logging

        # Only refine if there's feedback from a previous attempt (i.e., iteration > 1)
        # The check `iteration > 1` is handled by the caller (_build_generation_prompt)
        if not (validation_error or evaluation_feedback):
            # This case should ideally not be reached if called correctly, but safety first.
            print("Warning: _generate_refined_task called without feedback.")
            return None

        # Determine the original request context AND enhancement separately
        # The initial_goal is always based on the final_command (default or user-provided)
        initial_goal = state.get("final_command", agent_cfg.DEFAULT_GENERATOR_FINAL_COMMAND)
        enhancement_request = state.get("enhancement_request")  # Will be None if not enhancing

        summarizer_prompt = f"""
You are an expert prompt engineer assisting in a multi-turn AI workflow. Your goal is to refine the final 'Task' instruction for a Manim code generation AI based on the outcome of its previous attempt, considering the initial goal and any requested enhancements.

This refined 'Task' instruction will be placed at the *very end* of the main prompt given to the code generation AI. It's crucial that this instruction is clear, actionable, and leverages the context the code generation AI has already received.

**Context Provided to Code Generation AI (Before the Task You Generate):**
1. Manim programming rules and best practices (MANIM_RULES).
2. Relevant Manim documentation excerpts (`context_doc`).
3. General context about the overall video/animation goal (`general_context`).
4. (If applicable) The previous code attempt that failed or is being enhanced (`previous_code_attempt` or `generated_output` from the failed run).
5. (If applicable) Specific feedback detailing the validation error (`validation_error`).
6. (If applicable) Specific feedback detailing evaluation results based on a rubric (`evaluation_feedback`).
7. (If applicable) The specific text detailing the enhancement request.

**Your Input:**
- Initial Goal: "{initial_goal}"
- Enhancement Request (if applicable): "{enhancement_request or 'None'}"
- Validation Error (from previous attempt, if any): "{validation_error or 'None'}"
- Evaluation Feedback (from previous attempt, if any): "{evaluation_feedback or 'None'}"

**Your Task:**
Generate ONLY the text content for the final 'Task' instruction. Apply strong prompt engineering principles:
- Synthesize the 'Initial Goal' and the 'Enhancement Request' (if provided) to understand the *current* desired outcome.
- Be specific about the *fixes* needed, directly referencing the key issues from the validation error and/or evaluation feedback.
- Clearly reiterate the core objective (incorporating the enhancement if applicable).
- Ensure the instruction is concise and guides the code generation AI effectively on its *next* attempt to achieve the desired outcome while fixing the errors.
- Do **not** include the `--- Task ---` or `--- End Task ---` markers in your output. Just provide the instruction text itself.
"""

        log_run_details(
            run_output_dir, iteration, node_name, "Summarizer LLM Prompt", summarizer_prompt
        )

        try:
            print(f"Calling Task Summarizer LLM: {self.llm_text_client.model}")
            response = self.llm_text_client.invoke(summarizer_prompt)
            refined_task = response.content.strip()
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Summarizer LLM Response (Refined Task)",
                refined_task,
            )
            # Basic check: Ensure response isn't empty or just whitespace
            if refined_task:
                return refined_task
            else:
                print("WARNING: Task Summarizer LLM returned empty content. Falling back.")
                log_run_details(
                    run_output_dir,
                    iteration,
                    node_name,
                    "Summarizer LLM Warning",
                    "LLM returned empty content.",
                    is_error=True,
                )
                return None  # Fallback signal

        except Exception as e:
            error_message = f"Error calling Task Summarizer LLM: {e}"
            print(f"WARNING: {error_message}")
            log_run_details(
                run_output_dir,
                iteration,
                node_name,
                "Summarizer LLM Error",
                error_message,
                is_error=True,
            )
            return None  # Fallback signal

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
        thought_history = state.get("thought_history", [])[:]  # Initialize thought history

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
            "thought_history": thought_history,  # Pass copy to modify
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
            # Append error marker to thoughts history even on LLM error
            updates_to_state["thought_history"].append(
                f"Iteration {current_iteration_number}: LLM Error - {error_message}"
            )
            return updates_to_state

        # --- Parse Code and Thoughts ---
        log_run_details(
            run_output_dir,
            current_iteration_number,
            node_name,
            "Parsing",
            "Parsing LLM response for code and thoughts...",
        )
        extracted_code, extracted_thoughts = self._extract_code_and_thoughts(llm_output)

        # --- Update Thought History ---
        # Log thoughts whether they exist or not, and regardless of code validity
        thought_log_entry = f"Iteration {current_iteration_number}: {extracted_thoughts or 'No thoughts block found.'}"
        updates_to_state["thought_history"].append(thought_log_entry)
        log_run_details(
            run_output_dir,
            current_iteration_number,
            node_name,
            "Extracted Thoughts",
            thought_log_entry,
        )

        # --- Validate Code ---
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
            # Note: thought_history already updated above
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
            # Append validation failure info to the thought added earlier
            updates_to_state["thought_history"][
                -1
            ] += f" (Result: Failed Validation - {error_message})"
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
            # Append validation failure info to the thought added earlier
            updates_to_state["thought_history"][
                -1
            ] += f" (Result: Failed Validation - {error_message})"
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
            # Append syntax error info to the thought added earlier
            updates_to_state["thought_history"][
                -1
            ] += f" (Result: Failed Syntax Check - {error_message})"
            return updates_to_state

        # --- Update State ---
        log_run_details(
            run_output_dir,
            current_iteration_number,
            node_name,
            "Node Completion",
            "Code generation and basic validation successful.",
        )
        # Append success marker to the thought added earlier
        updates_to_state["thought_history"][-1] += " (Result: Passed Basic Validation)"
        updates_to_state["validation_error"] = None  # Clear validation error on success

        return updates_to_state
