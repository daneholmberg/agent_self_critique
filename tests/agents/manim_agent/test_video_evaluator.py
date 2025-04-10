import unittest
import os
import base64
from unittest.mock import patch, MagicMock, mock_open

from agents.manim_agent.video_evaluator import ManimVideoEvaluator
from core.graph_state import GraphState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Mock config values
class MockAgentConfig:
    VIDEO_EVAL_MAX_SIZE_MB = 50  # Example limit


class MockBaseConfig:
    BASE_DIR = "/fake/project/root"


# Patch config modules used by the evaluator
@patch("agents.manim_agent.video_evaluator.agent_cfg", MockAgentConfig)
@patch("agents.manim_agent.video_evaluator.base_cfg", MockBaseConfig)
class TestManimVideoEvaluator(unittest.TestCase):

    def setUp(self):
        """Set up a mock LLM client and the evaluator instance."""
        self.mock_llm_client = MagicMock(spec=ChatGoogleGenerativeAI)
        self.mock_llm_client.model = "mock-gemini-vision-model"
        self.evaluator = ManimVideoEvaluator(llm_eval_client=self.mock_llm_client)
        self.video_rel_path = "media/videos/output/SomeScene.mp4"
        self.video_full_path = os.path.join(MockBaseConfig.BASE_DIR, self.video_rel_path)
        self.initial_state: GraphState = {
            "input_text": "Make a square turn into a circle.",
            "rubric": "1. Is there a square?\n2. Does it become a circle?",
            "validated_artifact_path": self.video_rel_path,
            "iteration": 2,
            "error_history": [],
            "evaluation_history": [],
            # Add other required keys from GraphState with default values if needed
            "input_metadata": None,
            "max_iterations": 5,
            "generated_output": "some code",
            "validation_error": None,
            "evaluation_feedback": None,
            "evaluation_passed": None,
        }

    def test_build_evaluation_prompt(self):
        """Tests that the evaluation prompt is structured correctly."""
        prompt = self.evaluator._build_evaluation_prompt(self.initial_state)
        self.assertIn("Please evaluate the provided Manim video", prompt)
        self.assertIn("Rubric:", prompt)
        self.assertIn(self.initial_state["rubric"], prompt)
        self.assertIn("Original Script Segment:", prompt)
        self.assertIn(self.initial_state["input_text"], prompt)
        self.assertIn("Overall Assessment: PASS", prompt)
        self.assertIn("Overall Assessment: FAIL", prompt)

    @patch("builtins.open", new_callable=mock_open, read_data=b"videocontent")
    @patch("agents.manim_agent.video_evaluator.mimetypes.guess_type")
    @patch("agents.manim_agent.video_evaluator.base64.b64encode")
    def test_prepare_video_payload_success(self, mock_b64encode, mock_guess_type, mock_open_file):
        """Tests successful preparation of video payload."""
        mock_guess_type.return_value = ("video/mp4", None)
        mock_b64encode.return_value = (
            b"dmjefGVvY29udGVudA=="  # Mock base64 string without the space
        )

        payload = self.evaluator._prepare_video_payload(self.video_full_path)

        mock_open_file.assert_called_once_with(self.video_full_path, "rb")
        mock_b64encode.assert_called_once_with(b"videocontent")
        mock_guess_type.assert_called_once_with(self.video_full_path)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload), 1)
        item = payload[0]
        self.assertEqual(item["type"], "image_url")
        self.assertIn("url", item["image_url"])
        self.assertTrue(item["image_url"]["url"].startswith("data:video/mp4;base64,"))
        self.assertEqual(item["image_url"]["url"].split(",")[1], "dmjefGVvY29udGVudA==")

    @patch("builtins.open", side_effect=OSError("File not found"))
    def test_prepare_video_payload_file_error(self, mock_open_file):
        """Tests error handling when video file cannot be opened."""
        payload = self.evaluator._prepare_video_payload(self.video_full_path)
        mock_open_file.assert_called_once_with(self.video_full_path, "rb")
        self.assertIsNone(payload)

    @patch("agents.manim_agent.video_evaluator.os.path.exists")
    def test_evaluate_manim_video_no_artifact(self, mock_exists):
        """Tests evaluation skip when no validated artifact path exists."""
        no_artifact_state = self.initial_state.copy()
        no_artifact_state["validated_artifact_path"] = None

        result_state = self.evaluator.evaluate_manim_video(no_artifact_state)

        mock_exists.assert_not_called()
        self.mock_llm_client.invoke.assert_not_called()
        self.assertFalse(result_state["evaluation_passed"])
        self.assertIn("No validated video artifact path found", result_state["evaluation_feedback"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn(
            "Iter 2: Evaluation skipped: No validated video artifact path found",
            result_state["error_history"][0],
        )

    @patch("agents.manim_agent.video_evaluator.os.path.exists")
    def test_evaluate_manim_video_artifact_file_missing(self, mock_exists):
        """Tests evaluation skip when artifact path points to a non-existent file."""
        mock_exists.return_value = False  # File does not exist at the path

        result_state = self.evaluator.evaluate_manim_video(self.initial_state)

        mock_exists.assert_called_once_with(self.video_full_path)
        self.mock_llm_client.invoke.assert_not_called()
        self.assertFalse(result_state["evaluation_passed"])
        self.assertIn("No valid video file found", result_state["evaluation_feedback"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn(
            "Iter 2: Evaluation skipped: No valid video file found",
            result_state["error_history"][0],
        )

    @patch("agents.manim_agent.video_evaluator.os.path.exists", return_value=True)
    @patch("agents.manim_agent.video_evaluator.os.path.getsize")
    def test_evaluate_manim_video_too_large(self, mock_getsize, mock_exists):
        """Tests evaluation skip when video file exceeds size limit."""
        # Size = 60 MB > 50 MB limit
        mock_getsize.return_value = MockAgentConfig.VIDEO_EVAL_MAX_SIZE_MB * 1024 * 1024 + 1

        result_state = self.evaluator.evaluate_manim_video(self.initial_state)

        mock_exists.assert_called_once_with(self.video_full_path)
        mock_getsize.assert_called_once_with(self.video_full_path)
        self.mock_llm_client.invoke.assert_not_called()
        self.assertFalse(result_state["evaluation_passed"])
        self.assertIn("Video file too large", result_state["evaluation_feedback"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn(
            "Iter 2: Evaluation skipped: Video file too large", result_state["error_history"][0]
        )

    @patch("agents.manim_agent.video_evaluator.os.path.exists", return_value=True)
    @patch("agents.manim_agent.video_evaluator.os.path.getsize", return_value=1024 * 1024)
    @patch.object(ManimVideoEvaluator, "_prepare_video_payload")  # Mock the helper method
    def test_evaluate_manim_video_payload_error(
        self, mock_prepare_payload, mock_getsize, mock_exists
    ):
        """Tests evaluation skip when video payload preparation fails."""
        mock_prepare_payload.return_value = None  # Simulate failure

        result_state = self.evaluator.evaluate_manim_video(self.initial_state)

        mock_exists.assert_called_once_with(self.video_full_path)
        mock_getsize.assert_called_once_with(self.video_full_path)
        mock_prepare_payload.assert_called_once_with(self.video_full_path)
        self.mock_llm_client.invoke.assert_not_called()
        self.assertFalse(result_state["evaluation_passed"])
        self.assertIn("Failed to prepare video payload", result_state["evaluation_feedback"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn("Iter 2: Video Payload Error", result_state["error_history"][0])

    @patch("agents.manim_agent.video_evaluator.os.path.exists", return_value=True)
    @patch("agents.manim_agent.video_evaluator.os.path.getsize", return_value=1024 * 1024)
    @patch.object(ManimVideoEvaluator, "_prepare_video_payload")  # Mock the helper method
    def test_evaluate_manim_video_llm_error(self, mock_prepare_payload, mock_getsize, mock_exists):
        """Tests handling of LLM API errors during evaluation."""
        # Simulate successful payload prep
        mock_payload = [{"type": "image_url", "image_url": {"url": "data:video/mp4;base64,abc"}}]
        mock_prepare_payload.return_value = mock_payload

        # Simulate LLM API error
        self.mock_llm_client.invoke.side_effect = Exception("Quota exceeded")

        result_state = self.evaluator.evaluate_manim_video(self.initial_state)

        mock_exists.assert_called_once_with(self.video_full_path)
        mock_getsize.assert_called_once_with(self.video_full_path)
        mock_prepare_payload.assert_called_once_with(self.video_full_path)
        self.mock_llm_client.invoke.assert_called_once()
        self.assertFalse(result_state["evaluation_passed"])
        self.assertIn("Evaluation LLM Error", result_state["evaluation_feedback"])
        self.assertIn("Quota exceeded", result_state["evaluation_feedback"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn(
            "Iter 2: Evaluation LLM Error: Gemini API Error during evaluation: Quota exceeded",
            result_state["error_history"][0],
        )

    @patch("agents.manim_agent.video_evaluator.os.path.exists", return_value=True)
    @patch("agents.manim_agent.video_evaluator.os.path.getsize", return_value=1024 * 1024)
    @patch.object(ManimVideoEvaluator, "_prepare_video_payload")  # Mock the helper method
    def test_evaluate_manim_video_pass(self, mock_prepare_payload, mock_getsize, mock_exists):
        """Tests successful evaluation resulting in a PASS."""
        mock_payload = [
            {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,pass_video"}}
        ]
        mock_prepare_payload.return_value = mock_payload

        # Simulate LLM response indicating PASS
        mock_llm_response = MagicMock()
        mock_llm_response.content = "The video shows a square smoothly transforming into a circle. Looks good.\nOverall Assessment: PASS"
        self.mock_llm_client.invoke.return_value = mock_llm_response

        result_state = self.evaluator.evaluate_manim_video(self.initial_state)

        mock_exists.assert_called_once_with(self.video_full_path)
        mock_getsize.assert_called_once_with(self.video_full_path)
        mock_prepare_payload.assert_called_once_with(self.video_full_path)
        self.mock_llm_client.invoke.assert_called_once()
        # Check that the message passed to invoke has the correct structure
        args, kwargs = self.mock_llm_client.invoke.call_args
        self.assertEqual(len(args[0]), 1)  # Should be list containing one HumanMessage
        message_content = args[0][0].content
        self.assertEqual(len(message_content), 2)
        self.assertEqual(message_content[0]["type"], "text")
        self.assertEqual(message_content[1]["type"], "image_url")
        self.assertEqual(message_content[1]["image_url"]["url"], "data:video/mp4;base64,pass_video")

        self.assertTrue(result_state["evaluation_passed"])
        self.assertEqual(result_state["evaluation_feedback"], mock_llm_response.content)
        self.assertEqual(len(result_state["evaluation_history"]), 1)
        self.assertIn("Iter 2: PASS", result_state["evaluation_history"][0])
        self.assertIn(mock_llm_response.content, result_state["evaluation_history"][0])
        self.assertEqual(len(result_state["error_history"]), 0)

    @patch("agents.manim_agent.video_evaluator.os.path.exists", return_value=True)
    @patch("agents.manim_agent.video_evaluator.os.path.getsize", return_value=1024 * 1024)
    @patch.object(ManimVideoEvaluator, "_prepare_video_payload")  # Mock the helper method
    def test_evaluate_manim_video_fail(self, mock_prepare_payload, mock_getsize, mock_exists):
        """Tests successful evaluation resulting in a FAIL."""
        mock_payload = [
            {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,fail_video"}}
        ]
        mock_prepare_payload.return_value = mock_payload

        # Simulate LLM response indicating FAIL
        mock_llm_response = MagicMock()
        mock_llm_response.content = (
            "The video only shows a square, it does not transform.\nOverall Assessment: FAIL"
        )
        self.mock_llm_client.invoke.return_value = mock_llm_response

        result_state = self.evaluator.evaluate_manim_video(self.initial_state)

        self.mock_llm_client.invoke.assert_called_once()
        self.assertFalse(result_state["evaluation_passed"])
        self.assertEqual(result_state["evaluation_feedback"], mock_llm_response.content)
        self.assertEqual(len(result_state["evaluation_history"]), 1)
        self.assertIn("Iter 2: FAIL", result_state["evaluation_history"][0])
        self.assertIn(mock_llm_response.content, result_state["evaluation_history"][0])
        self.assertEqual(len(result_state["error_history"]), 0)


if __name__ == "__main__":
    unittest.main()
