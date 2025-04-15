import re
import json
import logging
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
import asyncio

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

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
- Using CYAN, LIGHT_BLUE manim colors, it's not a defined color. Favor using ManimColor(HEX) to get the specific color you want over using the manim color name.
- Use ShowCreation, it's deprecated for Creation"""


# --- Define Pydantic model for structured output ---
class ManimGenerationOutput(BaseModel):
    """Pydantic model for structured output from the Manim code generation LLM."""

    thoughts: str = Field(
        description="Detailed thought process, plan, and reasoning for the generated code."
    )
    code: str = Field(description="The complete Python code for the Manim scene.")


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
        self.parser = JsonOutputParser(pydantic_object=ManimGenerationOutput)
        logger.info(
            f"ManimCodeGenerator initialized with LLM: {llm_text_client.__class__.__name__}"
        )

    async def _build_generation_prompt(self, state: ManimAgentState) -> str:
        """
        Constructs the prompt for the Manim code generation LLM call using ManimAgentState.
        Relies on `with_structured_output` to handle formatting instructions.
        """
        task_instruction = state["task_instruction"]
        context_doc = state.get("context_doc", "")
        failure_summaries = state.get("failure_summaries", [])
        attempt_number = state.get("attempt_number", 0)
        scene_name = state.get("scene_name", agent_cfg.GENERATED_SCENE_NAME)
        rubric = state.get("rubric", "")  # Get the rubric for this task
        reflection_history = state.get("reflection_history", [])  # NEW: Get reflection history
        latest_reflection_entry = reflection_history[-1] if reflection_history else None
        latest_reflection = (
            latest_reflection_entry.get("reflection") if latest_reflection_entry else None
        )

        is_initial_enhancement = (
            state.get("previous_code_attempt")
            and state.get("enhancement_request")
            and attempt_number == 0
        )
        enhancement_request = state.get("enhancement_request")

        general_context = state.get("general_context", agent_cfg.DEFAULT_GENERATOR_GENERAL_CONTEXT)

        # MODIFIED: Retrieve template, check if None/empty, then format
        final_command_template = state.get("final_command")  # Get value, might be None
        if not final_command_template:
            logger.debug("No specific final_command in state, using default template.")
            final_command_template = agent_cfg.DEFAULT_GENERATOR_FINAL_COMMAND

        # Ensure template is valid before formatting
        if final_command_template and isinstance(final_command_template, str):
            try:
                final_command = final_command_template.format(scene_name=scene_name)
            except KeyError as e:
                logger.error(
                    f"Error formatting final_command template: Missing key {e}. Template: '{final_command_template}'"
                )
                # Fallback: Use a basic command indicating the error but including the scene name
                final_command = f"# ERROR: Template formatting failed. Ensure class is {scene_name}.\nGenerate the Python code."
        else:
            logger.error(
                f"Final command template is invalid (None or not a string) even after checking default: {final_command_template}"
            )
            # Fallback: Basic command if template is fundamentally broken
            final_command = f"# ERROR: Invalid command template. Ensure class is {scene_name}.\nGenerate the Python code."

        prompt_lines = [
            MANIM_RULES,
            f"You are a Manim v{agent_cfg.MANIM_VERSION_TARGET} expert tasked with generating Python code for a Manim scene.",
            "Follow the instructions in the provided Manim documentation context precisely.",
            "\n--- Manim Documentation Context ---",
            context_doc,
            "--- End Manim Documentation Context ---",
        ]

        if general_context:
            prompt_lines.extend(
                [
                    "\n--- General Context ---",
                    general_context.strip(),
                    "--- End General Context ---\n",
                ]
            )

        # --- Combined Reflection and Failure History ---
        reflection_history = state.get("reflection_history", [])

        if failure_summaries or reflection_history:
            prompt_lines.append("\n--- History of Past Attempts ---")

            # Create a mapping of attempt numbers to their details
            attempt_details = {}

            # Populate with failure summaries - they are 0-indexed in the list but represent attempts 1, 2, 3...
            for i, summary in enumerate(failure_summaries):
                attempt_num = i + 1  # Convert 0-based index to 1-based attempt number
                if attempt_num not in attempt_details:
                    attempt_details[attempt_num] = {"failure": summary, "reflection": None}
                else:
                    attempt_details[attempt_num]["failure"] = summary

            # Find the latest reflection attempt number - these are already 1-based in the data
            latest_reflection_attempt = (
                max([entry.get("attempt_index", 0) for entry in reflection_history], default=0)
                if reflection_history
                else 0
            )

            # Populate with reflections, excluding the latest one only if we have latest_reflection
            for entry in reflection_history:
                attempt_num = entry.get("attempt_index")
                if attempt_num is not None:
                    # Only exclude the latest reflection if we'll show it separately
                    if latest_reflection and attempt_num == latest_reflection_attempt:
                        continue

                    if attempt_num not in attempt_details:
                        attempt_details[attempt_num] = {
                            "failure": None,
                            "reflection": entry.get("reflection"),
                        }
                    else:
                        attempt_details[attempt_num]["reflection"] = entry.get("reflection")

            # Output the combined history in order
            for attempt_num in sorted(attempt_details.keys()):
                details = attempt_details[attempt_num]
                history_line = f"Attempt {attempt_num}:"

                if details["failure"]:
                    history_line += f" Failure summary: {details['failure']}"
                    if details["reflection"]:
                        history_line += f". "

                if details["reflection"]:
                    history_line += f" Reflection: {details['reflection']}"

                prompt_lines.append(history_line)

            prompt_lines.append("--- End History of Past Attempts ---\n")

        # --- Special Callout for Latest Reflection ---
        if latest_reflection:
            prompt_lines.append("\n--- Reflection on Last Attempt ---")
            prompt_lines.append(latest_reflection)
            prompt_lines.append("--- End Reflection on Last Attempt ---\n")
        # --- End Special Callout ---

        previous_code_this_cycle = state.get(
            "code"
        )  # Code from the *immediately preceding* node run in *this* loop

        if is_initial_enhancement:
            original_code_for_enhancement = state.get("previous_code_attempt")
            if original_code_for_enhancement:
                prompt_lines.append("--- Original Code (For Enhancement) ---")
                prompt_lines.append(f"```python\n{original_code_for_enhancement}\n```")
                prompt_lines.append("--- End Original Code ---")
            else:
                logger.warning("Enhancement requested but no 'previous_code_attempt' provided.")

            prompt_lines.append("\n--- Requested Enhancements ---")
            prompt_lines.append(enhancement_request)
            prompt_lines.append("--- End Requested Enhancements ---")
            prompt_lines.append("\nPlease enhance the original code based on the request above.")

        elif attempt_number > 0:
            fail_header = (
                "Failed Enhancement Attempt" if enhancement_request else "Failed Code Attempt"
            )
            if previous_code_this_cycle:
                prompt_lines.append(f"\n--- {fail_header} (Attempt {attempt_number}) --- ")
                prompt_lines.append(f"```python\n{previous_code_this_cycle}\n```")
                prompt_lines.append(f"--- End {fail_header} ---")
            else:
                prompt_lines.append(
                    f"\n--- Failed Attempt {attempt_number} (No Code from Previous Cycle Available) --- "
                )

            # --- Detailed Error/Feedback ---
            execution_success = state.get("execution_success")
            evaluation_passed = state.get("evaluation_passed")

            logger.info(
                f"State values in prompt builder - execution_success: {execution_success}, evaluation_passed: {evaluation_passed}"
            )

            # Check for execution error first
            if execution_success is False:
                detailed_error = state.get("validation_error")
                if detailed_error:
                    prompt_lines.append("\n--- Detailed Error from Last Execution Attempt ---")
                    prompt_lines.append(detailed_error)
                    prompt_lines.append("--- End Detailed Error ---")
            # Only check evaluation if we know it failed AND execution didn't explicitly fail
            elif evaluation_passed is False and execution_success is not False:
                evaluation_result_dict = state.get("evaluation_result")
                if evaluation_result_dict is not None:
                    detailed_feedback = evaluation_result_dict.get("feedback")
                    if detailed_feedback:
                        prompt_lines.append(
                            "\n--- Detailed Feedback from Last Evaluation Attempt ---"
                        )
                        prompt_lines.append(detailed_feedback)
                        prompt_lines.append("--- End Detailed Feedback ---")
                else:
                    logger.warning("evaluation_passed is False but evaluation_result_dict is None")
            # --- End Detailed Error/Feedback ---

            if enhancement_request:
                prompt_lines.append("\n--- Enhancement Request (Ongoing) ---")
                prompt_lines.append(enhancement_request)
                prompt_lines.append("--- End Enhancement Request ---")

            prompt_lines.append(
                "Please analyze the failure summaries, the LATEST REFLECTION, the DETAILED ERROR/FEEDBACK MESSAGE above (if provided), and the previous code attempt (if shown)."
            )
            prompt_lines.append(
                "Generate corrected code based on the original task and any enhancement requests."
            )

        # --- Include Rubric for Guidance ---
        # Only show rubric if:
        # 1. It's the first run (attempt_number == 0), or
        # 2. There was an error in execution or evaluation failed
        # 3. But don't show it if we already have evaluation results with feedback
        show_rubric = False

        if rubric:
            # First run - always show rubric
            if attempt_number == 0:
                show_rubric = True
            # Error/failure cases - show rubric to help fix issues
            elif state.get("execution_success") is False or state.get("evaluation_passed") is False:
                # But don't show it if we have detailed evaluation feedback
                evaluation_result_dict = state.get("evaluation_result", {})
                detailed_feedback = evaluation_result_dict.get("feedback")
                if not detailed_feedback:
                    show_rubric = True

            if show_rubric:
                prompt_lines.append("\n--- Evaluation Rubric ---")
                prompt_lines.append(rubric)
                prompt_lines.append("--- End Evaluation Rubric ---\n")
                prompt_lines.append(
                    "Use the above rubric as a guide for generating code that will pass evaluation."
                )
        # -----------------------------------

        # --- Primary Task Instruction ---
        prompt_lines.extend(
            [
                "\n--- Primary Task Instruction ---",
                task_instruction,
                "--- End Primary Task Instruction ---\n",
            ]
        )

        # --- Final Command (Updated for JSON output) ---
        prompt_lines.append(final_command)  # Already formatted with scene_name

        # Add explicit JSON formatting instructions
        prompt_lines.append("\nResponse MUST be a valid JSON object with the following structure:")
        prompt_lines.append("")
        prompt_lines.append("{")
        prompt_lines.append(
            '  "thoughts": "Your detailed thought process and plan for the generated code",'
        )
        prompt_lines.append('  "code": "The complete Python code for the Manim scene"')
        prompt_lines.append("}")
        prompt_lines.append(
            "THIS IS OF THE UTMOST IMPORTANCE. IT MUST BE VALID JSON. NO language mixing, JUST VALID JSON."
        )

        prompt_lines.append(
            "\nGenerate the Manim code based on the provided context, history, and instructions. Remember that it HAS to be valid JSON. No pythonic triple quotes, just valid JSON."
        )

        return "\n".join(prompt_lines)

    def clean_json_response(self, content: str) -> str:
        """
        Remove markdown formatting and extract the JSON object from LLM response.

        Args:
            content: The raw LLM response string

        Returns:
            A cleaned string containing only the JSON content
        """
        # Remove markdown code block markers if present
        if "```json" in content and "```" in content.split("```json", 1)[1]:
            # Extract content between ```json and the next ```
            content = content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            # Handle case where json keyword might be missing
            content = content.split("```", 1)[1].split("```", 1)[0].strip()

        # Ensure the content starts with a curly brace
        if not content.strip().startswith("{"):
            # Try to extract content between curly braces as fallback
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)

        return content

    async def generate_manim_code(self, state: ManimAgentState) -> Dict[str, Any]:
        """
        Generates Manim code based on the current state, using JSON output parsing.
        Uses JsonOutputParser with the Pydantic model ManimGenerationOutput for robust parsing.
        Updates the state with the generated code and generation history.

        Args:
            state: The current ManimAgentState.

        Returns:
            A dictionary containing the updated state fields: 'code', 'generation_history'.
            Returns {'error_message': ...} if generation fails critically.
        """
        logger.info("Starting Manim code generation using JSON parsing...")
        attempt_number = state.get("attempt_number", 0)  # Get current attempt number
        node_name = "ManimCodeGenerator"  # Define node_name for logging
        run_output_dir = state.get("run_output_dir", ".")  # Extract run_output_dir from state

        # --- Log Node Entry ---
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=attempt_number + 1,
            node_name=node_name,
            log_category="Node Entry",
            content=f"Starting {node_name} - Attempt {attempt_number + 1}",
        )
        # ----------------------

        # --- Initialize Histories if they don't exist ---
        generation_history = state.get("generation_history", [])
        reflection_history = state.get(
            "reflection_history", []
        )  # Ensure it exists, though we don't modify it here

        try:
            # 1. Build the prompt
            prompt = await self._build_generation_prompt(state)
            logger.debug(
                f"Generation prompt created (Attempt {attempt_number + 1}): {prompt[:500]}..."
            )  # Log start of prompt

            # --- Log the full prompt before sending ---
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=attempt_number + 1,
                node_name=node_name,
                log_category="LLM Prompt",
                content=f"Generator prompt for attempt {attempt_number + 1}:\n---\n{prompt}\n---",
            )
            # -------------------------------------------

            # 2. Call the LLM expecting the JSON output
            # Add timeout to prevent hanging
            try:
                # Create a task with an explicit timeout
                llm_task = asyncio.create_task(self.llm_text_client.ainvoke(prompt))
                llm_response = await asyncio.wait_for(llm_task, timeout=180)  # 3 minute timeout
                logger.info(
                    f"LLM response received within timeout for attempt {attempt_number + 1}"
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"LLM request timed out after 180 seconds for attempt {attempt_number + 1}"
                )
                return {
                    "error_message": "LLM request timed out after 180 seconds. Please try again.",
                    "generation_history": generation_history,
                    "reflection_history": reflection_history,
                    "code": None,
                }

            # Get the content from the response
            content = (
                llm_response.content if hasattr(llm_response, "content") else str(llm_response)
            )

            if not content:
                logger.error(f"LLM returned empty content for attempt {attempt_number + 1}")
                return {
                    "error_message": "LLM returned empty content. Please try again.",
                    "generation_history": generation_history,
                    "reflection_history": reflection_history,
                    "code": None,
                }

            # Log the raw response
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=attempt_number + 1,
                node_name=node_name,
                log_category="LLM Raw Response",
                content=f"Raw LLM Response: {content}",
            )

            # Clean and preprocess the response
            cleaned_content = self.clean_json_response(content)
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=attempt_number + 1,
                node_name=node_name,
                log_category="Preprocessed Response",
                content=f"Preprocessed response for parsing: {cleaned_content}...",
            )

            # 3. Parse the response using JsonOutputParser
            try:
                log_run_details(
                    run_output_dir=run_output_dir,
                    attempt_number=attempt_number + 1,
                    node_name=node_name,
                    log_category="Parse Result",
                    content="Attempting to parse JSON output using JsonOutputParser.",
                )
                # Parse the cleaned content instead of raw content
                parsed_result = self.parser.parse(cleaned_content)

                # Check the type of the parsed result
                if isinstance(parsed_result, ManimGenerationOutput):
                    logger.info(
                        f"Generation result (Pydantic): Thoughts length={len(parsed_result.thoughts)}, Code length: {len(parsed_result.code)}"
                    )
                    extracted_thoughts = parsed_result.thoughts
                    extracted_code = parsed_result.code
                elif isinstance(parsed_result, dict):
                    logger.warning(
                        f"Parsed result as dict (Pydantic validation might have partially failed): {parsed_result.keys()}"
                    )
                    # Attempt to access keys, default to empty/error message if missing
                    extracted_thoughts = parsed_result.get("thoughts", "No thoughts provided")
                    extracted_code = parsed_result.get("code")

                    # Log potentially missing keys
                    if "thoughts" not in parsed_result or "code" not in parsed_result:
                        logger.error(
                            f"Parsed dict missing 'thoughts' or 'code' key: {parsed_result}"
                        )
                        log_run_details(
                            run_output_dir=run_output_dir,
                            attempt_number=attempt_number + 1,
                            node_name=node_name,
                            log_category="Parse Warning",
                            content=f"Parsed dict missing 'thoughts' or 'code' key: {parsed_result}",
                            is_error=True,
                        )

                    # Check specifically for missing code
                    if not extracted_code:
                        logger.error("Parsed response missing 'code' field")
                        raise ValueError("Parsed response missing 'code' field")
                else:
                    # This case should ideally not happen if parser works as expected
                    raise TypeError(f"Unexpected parsed result type: {type(parsed_result)}")

                # Log the structured content
                log_run_details(
                    run_output_dir=run_output_dir,
                    attempt_number=attempt_number + 1,
                    node_name=node_name,
                    log_category="Parsed Content",
                    content=f"Parsed content for attempt {attempt_number + 1}:\n\n--- Thoughts ---\n{extracted_thoughts}\n\n--- Generated Code ---\n{extracted_code}",
                )

                # 4. Update Generation History
                new_generation_entry = {
                    "attempt_index": attempt_number + 1,  # History uses 1-based index
                    "thoughts": extracted_thoughts,
                    "code": extracted_code,
                }
                generation_history.append(new_generation_entry)

                # 5. Prepare state update
                update_dict = {
                    "code": extracted_code,
                    "generation_history": generation_history,
                    # Pass reflection history through unchanged
                    "reflection_history": reflection_history,
                    # Clear any previous error message specific to this node
                    "error_message": None,
                }

                # --- Log Node Success ---
                log_run_details(
                    run_output_dir=run_output_dir,
                    attempt_number=attempt_number + 1,
                    node_name=node_name,
                    log_category="Node Completion",
                    content=f"Code generation successful for attempt {attempt_number + 1}.",
                )
                # ------------------------

                logger.info(f"Code generation successful for attempt {attempt_number + 1}.")
                return update_dict

            except OutputParserException as pe:
                # Try fallback parsing if possible
                fallback_error = True
                fallback_dict = None

                try:
                    # Last resort: try standard json module
                    fallback_dict = json.loads(cleaned_content)
                    log_run_details(
                        run_output_dir=run_output_dir,
                        attempt_number=attempt_number + 1,
                        node_name=node_name,
                        log_category="Fallback Parsing",
                        content="Used fallback json.loads() parsing successfully",
                    )
                    fallback_error = False
                except json.JSONDecodeError:
                    pass

                if not fallback_error and fallback_dict:
                    # Handle successful fallback parsing
                    extracted_thoughts = fallback_dict.get("thoughts", "No thoughts provided")
                    extracted_code = fallback_dict.get("code")

                    if not extracted_code:
                        error_message = "Fallback parsing succeeded but 'code' field is missing"
                        logger.error(error_message)
                        log_run_details(
                            run_output_dir=run_output_dir,
                            attempt_number=attempt_number + 1,
                            node_name=node_name,
                            log_category="Parse Error",
                            content=error_message,
                            is_error=True,
                        )
                    else:
                        # Success with fallback parsing
                        log_run_details(
                            run_output_dir=run_output_dir,
                            attempt_number=attempt_number + 1,
                            node_name=node_name,
                            log_category="Fallback Success",
                            content=f"Successfully parsed with fallback method. Code length: {len(extracted_code)}",
                        )

                        # Update generation history
                        new_generation_entry = {
                            "attempt_index": attempt_number + 1,
                            "thoughts": extracted_thoughts,
                            "code": extracted_code,
                        }
                        generation_history.append(new_generation_entry)

                        return {
                            "code": extracted_code,
                            "generation_history": generation_history,
                            "reflection_history": reflection_history,
                            "error_message": None,
                        }
                else:
                    error_message = f"Failed to parse LLM response using all methods: {pe}\nRaw Response: {content[:500]}..."
                    logger.error(error_message)
                    log_run_details(
                        run_output_dir=run_output_dir,
                        attempt_number=attempt_number + 1,
                        node_name=node_name,
                        log_category="Parse Error",
                        content=error_message,
                        is_error=True,
                    )
                    return {
                        "error_message": error_message,
                        "generation_history": generation_history,
                        "reflection_history": reflection_history,
                        "code": None,
                    }
            except (ValueError, TypeError) as e:
                error_message = f"Error processing parsed result: {e}"
                logger.error(error_message)
                log_run_details(
                    run_output_dir=run_output_dir,
                    attempt_number=attempt_number + 1,
                    node_name=node_name,
                    log_category="Parse Error",
                    content=error_message,
                    is_error=True,
                )
                return {
                    "error_message": error_message,
                    "generation_history": generation_history,
                    "reflection_history": reflection_history,
                    "code": None,
                }

        except Exception as e:  # Catch any other unexpected errors
            # --- Log Node Error ---
            error_message_detail = f"Attempt {attempt_number + 1} failed: {e}"
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=attempt_number + 1,
                node_name=node_name,
                log_category="Generator Error",
                content=error_message_detail,
                is_error=True,
            )
            # ----------------------

            logger.exception(
                f"An unexpected error occurred during code generation (Attempt {attempt_number + 1}): {e}",
                exc_info=True,
            )
            error_message = f"Code generation failed: {e}"

            return {
                "error_message": error_message,
                # Pass histories back
                "generation_history": generation_history,
                "reflection_history": reflection_history,
                "code": None,  # Ensure code is None on error
            }


# --- Helper function (Consider moving to utils if reused) ---
def extract_scene_name(code: str) -> Optional[str]:
    """Extracts the first Manim Scene class name from the code."""
    # Regex to find class definitions inheriting from Scene (or subclasses like MovingCameraScene)
    match = re.search(r"class\s+([\\w\\d_]+)\\s*\\(\\s*(?:\\w*\\.)?Scene\\s*\\):", code)
    if match:
        return match.group(1)
    # Fallback for other scene types if needed
    match = re.search(r"class\s+([\\w\\d_]+)\\s*\\(.*Scene.*\\):", code)
    if match:
        return match.group(1)
    logger.warning("Could not automatically extract Scene name from generated code.")
    return None  # Return None if no scene name found
