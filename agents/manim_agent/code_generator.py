import re
import logging
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel

from agents.manim_agent import config as agent_cfg
from agents.manim_agent.config import ManimAgentState
from core.log_utils import log_run_details

logger = logging.getLogger(__name__)

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

    def __init__(self, llm_text_client: BaseLanguageModel):
        """
        Initializes the code generator with a text generation LLM client.

        Args:
            llm_text_client: The pre-initialized Langchain LLM client for text generation.
        """
        if not llm_text_client:
            raise ValueError("LLM client cannot be None for ManimCodeGenerator")
        self.llm_text_client = llm_text_client
        logger.info(
            f"ManimCodeGenerator initialized with LLM: {llm_text_client.__class__.__name__}"
        )

    @staticmethod
    def _extract_code_and_thoughts(text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts Python code block (```python ... ```) and thoughts block
        (```thoughts ... ```) from a string.
        Handles cases where fences might be missing.
        """
        # Use non-greedy matching (.+?) to handle multiple blocks better
        code_match = re.search(r"```python\n(.+?)```", text, re.DOTALL)
        extracted_code = code_match.group(1).strip() if code_match else None

        thoughts_match = re.search(r"```thoughts\n(.+?)```", text, re.DOTALL)
        extracted_thoughts = thoughts_match.group(1).strip() if thoughts_match else None

        # Fallback for code if no ```python block found
        if not extracted_code:
            # If thoughts were found, assume code might be the rest of the text
            if thoughts_match:
                thoughts_block_text = thoughts_match.group(0)
                potential_code = text.replace(thoughts_block_text, "").strip()
                # Basic sanity check for Python code
                if "import " in potential_code and (
                    "class " in potential_code or "def " in potential_code
                ):
                    extracted_code = potential_code
                    logger.debug("Extracted code using fallback (found thoughts, took remainder).")
            # If no code block AND no thoughts block, maybe the whole thing is code?
            elif "import " in text and ("class " in text or "def " in text):
                extracted_code = text.strip()
                logger.debug(
                    "Extracted code using fallback (no python/thoughts blocks, took whole text)."
                )

        if not extracted_code:
            logger.warning("Could not extract Python code block from LLM response.")
        if not extracted_thoughts:
            logger.debug("Could not extract thoughts block from LLM response (this might be okay).")

        return extracted_code, extracted_thoughts

    async def _build_generation_prompt(self, state: ManimAgentState) -> str:
        """
        Constructs the prompt for the Manim code generation LLM call using ManimAgentState.
        """
        # --- Extract Core Information ---
        task_instruction = state["task_instruction"]  # Current primary instruction
        context_doc = state.get("context_doc", "")  # Manim Documentation
        failure_summaries = state.get("failure_summaries", [])  # NEW: Get failure history
        attempt_number = state.get(
            "attempt_number", 0
        )  # Use attempt_number (0-based index of *previous* attempts)
        scene_name = state.get(
            "scene_name", agent_cfg.GENERATED_SCENE_NAME
        )  # Get scene name from state

        # Check if this is the *start* of an enhancement request
        is_initial_enhancement = (
            state.get("previous_code_attempt")
            and state.get("enhancement_request")
            and attempt_number == 0  # Check attempt_number
        )
        enhancement_request = state.get("enhancement_request")  # Persistent across retries

        # Use defaults from config if not present in state
        general_context = state.get("general_context", agent_cfg.DEFAULT_GENERATOR_GENERAL_CONTEXT)
        final_command = state.get("final_command", agent_cfg.DEFAULT_GENERATOR_FINAL_COMMAND)

        # --- Build Prompt ---
        prompt_lines = [
            MANIM_RULES,
            f"You are a Manim v{agent_cfg.MANIM_VERSION_TARGET} expert tasked with generating Python code for a Manim scene.",  # Use configured version
            "Follow the instructions in the provided Manim documentation context precisely.",
            "\n--- Manim Documentation Context ---",
            context_doc,
            "--- End Manim Documentation Context ---",
        ]

        # Add General Context if provided
        if general_context:
            prompt_lines.extend(
                [
                    "\n--- General Context ---",
                    general_context.strip(),
                    "--- End General Context ---\n",
                ]
            )

        # --- NEW: Add Failure Summaries History ---
        if failure_summaries:
            prompt_lines.append("\n--- History of Failure Summaries for this Task ---")
            for i, summary in enumerate(failure_summaries):
                # Add 1 to attempt_number because it's 0-based for *previous* attempts
                prompt_lines.append(
                    f"Attempt {i + 1} Failure: {summary}"
                )  # Label with attempt number
            prompt_lines.append("--- End History of Failure Summaries ---\n")

        # --- Previous Code & Enhancement / Failure Feedback ---
        # Use state["code"] for the *immediately preceding* attempt's code, if it exists
        previous_code_this_cycle = state.get("code")

        if is_initial_enhancement:
            # First attempt of an enhancement request - uses 'previous_code_attempt' from input
            original_code_for_enhancement = state.get("previous_code_attempt")
            if original_code_for_enhancement:
                prompt_lines.append("--- Original Code (For Enhancement) ---")
                prompt_lines.append(f"```python\n{original_code_for_enhancement}\n```")
                prompt_lines.append("--- End Original Code ---")
            else:
                logger.warning(
                    "Enhancement requested but no 'previous_code_attempt' provided in initial state."
                )

            prompt_lines.append("\n--- Requested Enhancements ---")
            prompt_lines.append(enhancement_request)
            prompt_lines.append("--- End Requested Enhancements ---")
            prompt_lines.append("\nPlease enhance the original code based on the request above.")

        elif attempt_number > 0:  # Check attempt_number
            # This is a retry attempt (attempt_number > 0)
            if previous_code_this_cycle:
                fail_header = (
                    "Failed Enhancement Attempt" if enhancement_request else "Failed Code Attempt"
                )
                prompt_lines.append(
                    f"\n--- {fail_header} (Attempt {attempt_number}) --- "  # Use attempt_number
                )
                prompt_lines.append(f"```python\n{previous_code_this_cycle}\n```")
                prompt_lines.append(f"--- End {fail_header} ---")
            else:
                prompt_lines.append(
                    f"\n--- Failed Attempt {attempt_number} (No Code from Previous Cycle Available) --- "
                )

            # --- NEW: Add Detailed Error from Last Execution ---
            # Check if the previous step was specifically a failed execution
            if state.get("execution_success") is False:
                detailed_error = state.get("validation_error")
                if detailed_error:
                    prompt_lines.append("\n--- Detailed Error from Last Execution Attempt ---")
                    prompt_lines.append(detailed_error)
                    prompt_lines.append("--- End Detailed Error ---")
                else:
                    logger.warning(
                        f"Execution success is False for attempt {attempt_number}, but no detailed 'validation_error' found in state."
                    )
            # --- End Detailed Error Section ---

            # Add specific feedback based on whether it's an enhancement retry or not
            if enhancement_request:
                prompt_lines.append("\n--- Enhancement Request (Ongoing) ---")
                prompt_lines.append(enhancement_request)
                prompt_lines.append("--- End Enhancement Request ---")

            prompt_lines.append(
                "\nPlease analyze the failure summaries, the DETAILED ERROR MESSAGE above (if provided), and the previous code attempt (if shown), then generate corrected code based on the original task and any enhancement requests."
            )

        # --- End Previous Code / Feedback ---

        prompt_lines.append("\n--- Primary Task Instruction ---")
        prompt_lines.append(f'"""\n{task_instruction}\n"""')  # Use the core task instruction
        prompt_lines.append("--- End Primary Task Instruction ---")

        # --- Determine Final Action Command ---
        # Get the base command template (either default or user-provided)
        final_command_template = state.get(
            "final_command", agent_cfg.DEFAULT_GENERATOR_FINAL_COMMAND
        )

        task_content = None
        if attempt_number > 0:  # Check attempt_number
            # Adding 1 to attempt_number for logging the *upcoming* attempt number
            logger.info(f"Attempt {attempt_number + 1}, attempting to generate refined task.")
            # Call the helper to get the refined task
            # The refiner is instructed to include the scene name
            task_content = await self._generate_refined_task(state)

        # Fallback if refinement wasn't attempted or failed
        if not task_content:
            logger.info("Using base final command template (no refinement or refinement failed).")
            # Format the base template with the scene name from state
            try:
                task_content = final_command_template.format(scene_name=scene_name)
            except KeyError:
                # If the template doesn't have {scene_name}, use it as is but log a warning
                logger.warning(
                    f'Provided final_command template does not contain a {{scene_name}} placeholder. Using it as is: "{final_command_template}"'
                )
                # Add explicit instruction about scene name *after* the user's command
                task_content = f"{final_command_template}\n\nCRITICAL: Ensure the generated code defines a class named '{scene_name}'."
        else:
            # If refinement succeeded, task_content already contains the refined command
            # (which should include the scene name instruction from the refiner prompt)
            logger.info(f"Using refined task instruction: {task_content[:100]}...")

        # --- Append Final Action Command ---
        prompt_lines.append(f"\n--- Action Command --- ")
        prompt_lines.append(task_content.strip())
        prompt_lines.append("--- End Action Command ---")

        # Final instruction on output format
        prompt_lines.append(
            "\nGenerate the complete Python code for the scene, enclosed in ```python ... ``` markers."
            # " Additionally, provide your thought process, reflections on previous attempts (drawing from the failure summaries), and plan for this generation attempt in a separate ```thoughts ... ``` block. Focus on addressing the past failures." # Kept for potential future use
        )
        return "\n".join(prompt_lines)

    async def _generate_refined_task(self, state: ManimAgentState) -> Optional[str]:
        """
        Uses the LLM to generate a refined, specific task instruction based on
        previous errors or feedback.

        Args:
            state: The current ManimAgentState containing feedback.

        Returns:
            The refined task string, or None if no feedback exists or an error occurs.
        """
        node_name = "CodeGenerator_TaskRefiner"
        run_output_dir = Path(state["run_output_dir"])
        # current_attempt is the number for the upcoming attempt (1-based)
        current_attempt = state.get("attempt_number", 0) + 1

        logger.info("Attempting to generate refined task instruction.")
        validation_error = state.get("validation_error")
        evaluation_result = state.get("evaluation_result")
        evaluation_feedback = evaluation_result.get("reasoning") if evaluation_result else None
        failure_summaries = state.get("failure_summaries", [])
        scene_name = state.get("scene_name", agent_cfg.GENERATED_SCENE_NAME)  # Get scene name

        if not (validation_error or evaluation_feedback or failure_summaries):
            logger.warning("_generate_refined_task called without any feedback or summaries.")
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Warning",
                "_generate_refined_task called without any feedback or summaries. Skipping refinement.",
                is_error=True,
            )
            return None

        formatted_summaries = "\n".join([f"- {s}" for s in failure_summaries])
        if not formatted_summaries:
            formatted_summaries = "None"

        initial_goal = state.get("task_instruction", "[Task Instruction Missing]")
        enhancement_request = state.get("enhancement_request")

        summarizer_prompt = f"""
        You are an expert prompt engineer assisting in a multi-turn AI workflow. Your goal is to refine the final 'Action Command' instruction for a Manim code generation AI based on the outcome of its previous attempt(s), considering the initial goal and any requested enhancements.

        This refined 'Action Command' instruction will be placed at the *very end* of the main prompt given to the code generation AI. It's crucial that this instruction is clear, actionable, and leverages the context the code generation AI has already received (including Failure Summaries provided earlier in the prompt).

        **Context Provided to Code Generation AI (Before the Command You Generate):**
        1. Manim programming rules and best practices.
        2. Relevant Manim documentation excerpts.
        3. General context about the overall video/animation goal.
        4. A history of concise failure summaries from past attempts.
        5. The specific code from the *immediately preceding* failed attempt (if available).
        6. (If applicable) The specific text detailing any ongoing enhancement request.

        **Your Input for Refinement:**
        - Initial Primary Task Instruction: "{initial_goal}"
        - Enhancement Request (if applicable): "{enhancement_request or 'None'}"
        - Historical Failure Summaries:\n{formatted_summaries}
        - Validation Error (from LATEST attempt, if any): "{validation_error or 'None'}"
        - Evaluation Feedback (from LATEST attempt, if any): "{evaluation_feedback or 'None'}"

        **Your Task:**
        Generate ONLY the text content for the final 'Action Command' instruction. Apply strong prompt engineering principles:
        - Synthesize the 'Initial Primary Task Instruction' and the 'Enhancement Request' (if provided) to understand the *current* desired outcome.
        - Analyze the **Historical Failure Summaries** to identify any recurring issues or patterns.
        - Be specific about the *fixes* needed, directly referencing the key issues from the **latest** validation error and/or evaluation feedback, but also consider addressing any **persistent problems** revealed by the history. Focus on creating a single, actionable command for the *next* attempt.
        - Clearly reiterate the core objective (incorporating the enhancement if applicable).
        - **CRITICAL**: Ensure the generated command explicitly instructs the code generation AI to define a Python class named '{scene_name}'.
        - Ensure the instruction is concise and guides the code generation AI effectively on its *next* attempt to achieve the desired outcome while fixing the errors described (both latest and historical patterns).
        - Do **not** include the `--- Action Command ---` or `--- End Action Command ---` markers in your output. Just provide the instruction text itself.
        """

        refiner_prompt_log_path = run_output_dir / f"task_refiner_prompt_iter_{current_attempt}.txt"
        try:
            with open(refiner_prompt_log_path, "w", encoding="utf-8") as f:
                f.write(summarizer_prompt)
            logger.info(f"Task refiner prompt saved to: {refiner_prompt_log_path}")
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Refiner LLM Prompt",
                summarizer_prompt,
            )
        except Exception as log_e:
            logger.warning(f"Could not save task refiner prompt log: {log_e}")

        try:
            logger.info(f"Calling Task Refiner LLM: {self.llm_text_client.__class__.__name__}")
            response = await self.llm_text_client.ainvoke(
                summarizer_prompt, config={"configurable": {"temperature": 0.3}}
            )
            refined_task = (
                response.content.strip() if hasattr(response, "content") else str(response).strip()
            )

            refiner_resp_log_path = (
                run_output_dir / f"task_refiner_response_iter_{current_attempt}.txt"
            )
            try:
                with open(refiner_resp_log_path, "w", encoding="utf-8") as f:
                    f.write(refined_task)
                logger.info(f"Task refiner response saved to: {refiner_resp_log_path}")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Refiner LLM Response",
                    refined_task,
                )
            except Exception as log_e:
                logger.warning(f"Could not save task refiner response log: {log_e}")

            if refined_task:
                logger.info(f"Successfully generated refined task: {refined_task[:100]}...")
                return refined_task
            else:
                logger.warning("Task Refiner LLM returned empty content. Falling back.")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Refiner Warning",
                    "Task Refiner LLM returned empty content. Falling back to default command.",
                    is_error=True,
                )
                return None

        except Exception as e:
            logger.error(f"Error calling Task Refiner LLM: {e}", exc_info=True)
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Refiner LLM Error",
                f"Error: {e}",
                is_error=True,
            )
            return None

    async def generate_manim_code(self, state: ManimAgentState) -> Dict[str, Any]:
        """
        Generates Manim code based on the current state.

        Args:
            state: The current ManimAgentState.

        Returns:
            A dictionary containing the generated code under the key "code",
            or an error message under "error_message" or "validation_error".
        """
        node_name = "ManimCodeGenerator"
        run_output_dir = Path(state["run_output_dir"])
        # current_attempt is the number for *this* generation attempt (1-based)
        current_attempt = state.get("attempt_number", 0) + 1

        log_run_details(
            run_output_dir,
            current_attempt,
            node_name,
            "Node Entry",
            f"Starting {node_name} - Attempt {current_attempt}",
        )

        updates_to_state: Dict[str, Any] = {
            "code": None,
            "validation_error": None,
            "error_message": None,
            "run_output_dir": str(run_output_dir),
            "scene_name": state.get("scene_name"),
            "save_generated_code": state.get("save_generated_code"),
        }

        try:
            prompt = await self._build_generation_prompt(state)
            prompt_log_path = run_output_dir / f"code_gen_prompt_iter_{current_attempt}.txt"
            with open(prompt_log_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info(f"Generation prompt saved to: {prompt_log_path}")
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "LLM Prompt",
                prompt,
            )

            response = await self.llm_text_client.ainvoke(prompt)
            llm_response_text = response.content if hasattr(response, "content") else str(response)

            response_log_path = run_output_dir / f"code_gen_response_iter_{current_attempt}.txt"
            with open(response_log_path, "w", encoding="utf-8") as f:
                f.write(llm_response_text)
            logger.info(f"Generation response saved to: {response_log_path}")
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "LLM Response",
                llm_response_text,
            )

            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Parsing",
                "Parsing LLM response for code and thoughts...",
            )

            generated_code, llm_thoughts = self._extract_code_and_thoughts(llm_response_text)

            if not generated_code:
                error_message = (
                    "Failed to parse Python code block (```python ... ```) from LLM response."
                )
                logger.error(error_message)
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Parsing Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["error_message"] = error_message
                return updates_to_state

            scene_name = state.get("scene_name", agent_cfg.GENERATED_SCENE_NAME)
            expected_class_pattern = rf"class\s+{re.escape(scene_name)}\s*\(.*Scene.*\):"
            if not re.search(expected_class_pattern, generated_code):
                error_message = f"Generated code missing correct Scene class definition inheriting from Scene: expected similar to 'class {scene_name}(Scene):'."
                logger.error(error_message)
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Validation Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["validation_error"] = error_message
                return updates_to_state

            try:
                compile(generated_code, "<string>", "exec")
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Syntax Check",
                    "Passed basic Python syntax check.",
                )
            except SyntaxError as e:
                error_message = f"Generated code has SyntaxError: {e}"
                logger.error(error_message)
                log_run_details(
                    run_output_dir,
                    current_attempt,
                    node_name,
                    "Syntax Error",
                    error_message,
                    is_error=True,
                )
                updates_to_state["validation_error"] = error_message
                return updates_to_state

            logger.info(f"Successfully generated and validated code (attempt {current_attempt}).")
            updates_to_state["code"] = generated_code
            updates_to_state["validation_error"] = None
            updates_to_state["error_message"] = None

            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Node Completion",
                "Code generation and basic validation successful.",
            )

        except Exception as e:
            error_message = f"Unexpected error during Manim code generation: {e}"
            logger.error(error_message, exc_info=True)
            log_run_details(
                run_output_dir,
                current_attempt,
                node_name,
                "Generator Error",
                error_message,
                is_error=True,
            )
            updates_to_state["validation_error"] = f"[Generator Internal Error] {error_message}"

        return updates_to_state
