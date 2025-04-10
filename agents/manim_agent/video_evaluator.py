import os
import re
import base64
import mimetypes
from typing import Dict, Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from config import base_config as base_cfg
from agents.manim_agent import config as agent_cfg
from core.graph_state import GraphState


class ManimVideoEvaluator:
    """Handles the evaluation of generated Manim videos using a multimodal LLM."""

    def __init__(self, llm_eval_client: ChatGoogleGenerativeAI):
        """
        Initializes the video evaluator with a multimodal LLM client.

        Args:
            llm_eval_client: The pre-initialized Langchain LLM client capable of video evaluation.
        """
        self.llm_eval_client = llm_eval_client

    def _build_evaluation_prompt(self, state: GraphState) -> str:
        """Constructs the prompt text for the video evaluation LLM call."""
        input_text = state["input_text"]
        rubric = state["rubric"]
        # Ensure the prompt clearly asks for the specific PASS/FAIL string
        return f"""Please evaluate the provided Manim video based on the following criteria (rubric) and the original script segment it visualizes.

Rubric:
{rubric}

Original Script Segment:
{input_text}

Provide a detailed evaluation addressing how well the video meets each point in the rubric.
Conclude your evaluation with the exact phrase 'Overall Assessment: PASS' if the video adequately visualizes the script according to the rubric, or 'Overall Assessment: FAIL' otherwise.
"""

    def _prepare_video_payload(self, full_video_path: str) -> Optional[List[Dict]]:
        """Reads, encodes, and formats the video file for multimodal LLM input.
        Assumes the file exists and is within size limits (checked by caller).

        Args:
            full_video_path: The absolute path to the video file.

        Returns:
            A list containing the formatted video payload dict for Langchain, or None on error.
        """
        try:
            print(f"Preparing video payload for: {full_video_path}")
            with open(full_video_path, "rb") as video_file:
                video_bytes = video_file.read()

            video_base64 = base64.b64encode(video_bytes).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(full_video_path)
            if not mime_type:
                mime_type = "video/mp4"  # Sensible default
                print(
                    f"WARN: Could not guess MIME type for {full_video_path}, defaulting to {mime_type}"
                )

            # Structure required by Langchain for multimodal input
            return [
                {
                    "type": "image_url",  # Langchain uses "image_url" for multimodal data
                    "image_url": {"url": f"data:{mime_type};base64,{video_base64}"},
                }
            ]
        except OSError as e:
            print(f"ERROR: OS error preparing video payload for {full_video_path}: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Unexpected error preparing video payload for {full_video_path}: {e}")
            return None

    def evaluate_manim_video(self, state: GraphState) -> Dict:
        """Evaluates the generated Manim video using the LLM based on the rubric."""
        print("---EVALUATE MANIM VIDEO NODE---")

        # 1. Initialize
        evaluation_history = state.get("evaluation_history", [])[:]
        error_history = state.get("error_history", [])[:]  # Copy in case we add pre-eval errors
        updates_to_state: Dict = {
            "evaluation_feedback": None,
            "evaluation_passed": None,
            "evaluation_history": evaluation_history,  # Pass copy
            "error_history": error_history,  # Pass copy
        }
        current_iteration = state.get("iteration", "N/A")

        # 2. Check Input Video Path (Relative Path)
        validated_artifact_relative_path = state.get("validated_artifact_path")
        if not validated_artifact_relative_path:
            error_message = "Evaluation skipped: No validated video artifact path found."
            print(f"WARN: {error_message}")
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False  # Explicitly fail if no video
            error_history.append(f"Iter {current_iteration}: {error_message}")
            return updates_to_state

        # Construct full path
        full_video_path = os.path.join(base_cfg.BASE_DIR, validated_artifact_relative_path)

        # Check existence
        if not os.path.exists(full_video_path):
            error_message = f"Evaluation skipped: No valid video file found at {full_video_path}."
            print(f"WARN: {error_message}")
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(f"Iter {current_iteration}: {error_message}")
            return updates_to_state

        # 3. Check Video Size
        try:
            video_size = os.path.getsize(full_video_path)
            max_size_bytes = agent_cfg.VIDEO_EVAL_MAX_SIZE_MB * 1024 * 1024
            if video_size > max_size_bytes:
                size_mb = video_size / (1024 * 1024)
                limit_mb = agent_cfg.VIDEO_EVAL_MAX_SIZE_MB
                error_message = (
                    f"Evaluation skipped: Video file too large ({size_mb:.2f} MB > {limit_mb} MB)."
                )
                print(f"WARN: {error_message}")
                updates_to_state["evaluation_feedback"] = error_message
                updates_to_state["evaluation_passed"] = False
                error_history.append(f"Iter {current_iteration}: {error_message}")
                return updates_to_state
        except OSError as e:
            error_message = (
                f"Evaluation skipped: Error checking video file size for {full_video_path}: {e}"
            )
            print(f"ERROR: {error_message}")
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(f"Iter {current_iteration}: {error_message}")
            return updates_to_state

        # 4. Prepare Multimodal Prompt Payload (Video Part)
        print(f"Preparing video artifact from validated path: {full_video_path}")
        video_payload = self._prepare_video_payload(full_video_path)

        if not video_payload:
            error_message = (
                f"Evaluation skipped: Failed to prepare video payload for {full_video_path}."
            )
            print(f"ERROR: {error_message}")
            updates_to_state["evaluation_feedback"] = error_message
            updates_to_state["evaluation_passed"] = False
            error_history.append(f"Iter {current_iteration}: Video Payload Error (see logs)")
            return updates_to_state

        # Prepare text part of the prompt
        evaluation_prompt_text = self._build_evaluation_prompt(state)

        # Structure the input message list correctly for ChatGoogleGenerativeAI
        message = HumanMessage(
            content=[
                {"type": "text", "text": evaluation_prompt_text},
                *video_payload,  # Unpack the list containing the video payload dict
            ]
        )

        # 5. Call Gemini Client (uses injected self.llm_eval_client)
        try:
            print(f"Calling Evaluation LLM: {self.llm_eval_client.model}")
            response = self.llm_eval_client.invoke([message])  # Pass message in a list
            llm_output = response.content
            print(f"Raw Evaluation Output:\n{llm_output}")
        except Exception as e:
            error_message = f"Gemini API Error during evaluation: {e}"
            print(f"ERROR: {error_message}")
            updates_to_state["evaluation_feedback"] = (
                f"Evaluation LLM Error: {e}"  # Concise state message
            )
            updates_to_state["evaluation_passed"] = False
            error_history.append(f"Iter {current_iteration}: Evaluation LLM Error: {error_message}")
            return updates_to_state

        # 6. Parse Feedback
        extracted_feedback = llm_output.strip()
        updates_to_state["evaluation_feedback"] = extracted_feedback

        # 7. Determine Pass/Fail
        # Search case-insensitive for the exact phrase 'Overall Assessment: PASS' at the end of the string might be safer
        evaluation_passed = bool(
            re.search(
                r"Overall Assessment: PASS\s*$", extracted_feedback, re.IGNORECASE | re.MULTILINE
            )
        )
        updates_to_state["evaluation_passed"] = evaluation_passed

        print(f"Evaluation Result: {'PASS' if evaluation_passed else 'FAIL'}")
        # Append result to evaluation history
        evaluation_history.append(
            f"Iter {current_iteration}: {'PASS' if evaluation_passed else 'FAIL'}\nFeedback:\n{extracted_feedback}"
        )
        # updates_to_state already contains the updated evaluation_history

        # 8. Return
        return updates_to_state
