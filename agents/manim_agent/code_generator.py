import re
from typing import Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState


class ManimCodeGenerator:
    """Handles the generation of Manim Python code using an LLM."""

    def __init__(
        self, llm_text_client: ChatGoogleGenerativeAI, script_context: Optional[str] = None
    ):
        """
        Initializes the code generator with a text generation LLM client and optional script context.

        Args:
            llm_text_client: The pre-initialized Langchain LLM client for text generation.
            script_context: Optional full script context provided by the user.
        """
        self.llm_text_client = llm_text_client
        self.script_context = script_context

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
        """Constructs the prompt for the Manim code generation LLM call."""
        input_text = state["input_text"]
        input_metadata = state.get("input_metadata")
        context_doc = state["context_doc"]
        # Rubric is not directly used in generation prompt, but passed in state

        prompt_lines = [
            f"You are a Manim v0.19 expert. Generate Python code for a Manim CE scene named '{agent_cfg.GENERATED_SCENE_NAME}'.",
            "Follow the instructions in the provided Manim documentation context precisely.",
            "The code should visualize the following script segment:",
            f'"""\n{input_text}\n"""',
        ]
        if input_metadata:
            prompt_lines.append(f"Consider this metadata: {input_metadata}")

        # Add the full script context if provided during initialization
        if self.script_context:
            prompt_lines.extend(
                [
                    "\n--- Full Script Context (for background understanding) ---",
                    self.script_context,
                    "--- End Full Script Context ---",
                ]
            )

        prompt_lines.extend(
            [
                "\n--- Manim Documentation Context ---",
                context_doc,
                "\n--- End Manim Documentation Context ---",
            ]
        )

        if iteration > 1:
            print(f"Retry {iteration-1}. Incorporating feedback...")
            prev_validation_error = state.get("validation_error")
            prev_eval_feedback = state.get("evaluation_feedback")

            if prev_validation_error or prev_eval_feedback:
                prompt_lines.append("\n--- Previous Attempt Feedback ---")
                if prev_validation_error:
                    prompt_lines.append(
                        "The previous code failed validation/execution with this error:"
                    )
                    prompt_lines.append(f'"""\n{prev_validation_error}\n"""')
                if prev_eval_feedback:
                    prompt_lines.append(
                        "The previous code received this evaluation feedback (address these points):"
                    )
                    prompt_lines.append(f'"""\n{prev_eval_feedback}\n"""')
                prompt_lines.append(
                    "Please fix the issues identified above in the new code you generate."
                )
                prompt_lines.append("--- End Previous Attempt Feedback ---")
            else:
                print("Warning: Iteration > 1 but no previous error or feedback found in state.")

        prompt_lines.append(
            "\nGenerate ONLY the complete Python code for the scene, enclosed in ```python ... ``` markers."
        )
        return "\n".join(prompt_lines)

    def generate_manim_code(self, state: GraphState) -> Dict:
        """Generates Manim Python code based on the input text and context using the LLM."""
        print("---GENERATE MANIM CODE NODE---")

        iteration = state["iteration"] + 1
        # Make copies of history lists to avoid modifying the original state directly
        error_history = state.get("error_history", [])[:]
        evaluation_history = state.get("evaluation_history", [])[
            :
        ]  # Keep track even if not used directly here

        updates_to_state: Dict = {
            "iteration": iteration,
            "generated_output": None,
            "validation_error": None,
            "validated_artifact_path": None,  # Reset previous artifact path
            "evaluation_feedback": None,  # Reset previous evaluation
            "evaluation_passed": None,  # Reset previous evaluation status
            "error_history": error_history,  # Pass copies
            "evaluation_history": evaluation_history,  # Pass copies
        }

        # --- Construct Prompt using helper ---
        prompt = self._build_generation_prompt(state, iteration)

        # --- Call LLM using injected client ---
        try:
            print(f"Calling Text Generation LLM: {self.llm_text_client.model}")
            response = self.llm_text_client.invoke(prompt)
            llm_output = response.content
        except Exception as e:
            error_message = f"Gemini API Error during code generation: {e}"
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            updates_to_state["error_history"].append(f"Iter {iteration}: {error_message}")
            return updates_to_state

        # --- Parse & Validate Code ---
        print("Parsing LLM response...")
        extracted_code = self._extract_python_code(llm_output)

        if not extracted_code:
            error_message = (
                "Failed to parse Python code block (```python ... ```) from LLM response."
            )
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            updates_to_state["error_history"].append(f"Iter {iteration}: {error_message}")
            return updates_to_state

        # Set the generated code in the state *before* validation checks
        updates_to_state["generated_output"] = extracted_code

        # Basic validation checks
        if "from manim import" not in extracted_code:
            error_message = "Generated code missing 'from manim import ...' statement."
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            updates_to_state["error_history"].append(f"Iter {iteration}: {error_message}")
            return updates_to_state

        expected_class_def = f"class {agent_cfg.GENERATED_SCENE_NAME}(Scene):"
        # Allow inheritance from subclasses of Scene as well
        if not re.search(
            rf"class\s+{agent_cfg.GENERATED_SCENE_NAME}\s*\(.*?Scene.*\):", extracted_code
        ):
            error_message = f"Generated code missing correct Scene class definition: expected '{expected_class_def}' or subclass."
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            updates_to_state["error_history"].append(f"Iter {iteration}: {error_message}")
            return updates_to_state

        # Syntax Check using compile()
        try:
            compile(extracted_code, "<string>", "exec")
            print("Generated code passed basic syntax check.")
        except SyntaxError as e:
            error_message = f"Generated code has SyntaxError: {e}"
            print(f"ERROR: {error_message}")
            updates_to_state["validation_error"] = error_message
            updates_to_state["error_history"].append(f"Iter {iteration}: {error_message}")
            return updates_to_state

        # --- Update State ---
        print("Code generation and basic validation successful.")
        updates_to_state["validation_error"] = None  # Explicitly clear potential previous error

        return updates_to_state
