import unittest
import os
import uuid
from unittest.mock import patch, MagicMock, mock_open, call

from agents.manim_agent.script_executor import ManimScriptExecutor
from core.graph_state import GraphState


# Mock config values
class MockAgentConfig:
    GENERATED_SCENE_NAME = "TestScene"
    MANIM_QUALITY_FLAG = "-ql"  # Example quality flag
    TEMP_SCRIPT_DIR = "./temp_scripts_test"
    MANIM_VIDEO_OUTPUT_RELATIVE_PATH = "media/videos/test_output"
    EXPECTED_VIDEO_FILENAME = "TestScene.mp4"


class MockBaseConfig:
    BASE_DIR = "/fake/project/root"


# Patch config modules used by the executor
@patch("agents.manim_agent.script_executor.agent_cfg", MockAgentConfig)
@patch("agents.manim_agent.script_executor.base_cfg", MockBaseConfig)
class TestManimScriptExecutor(unittest.TestCase):

    def setUp(self):
        """Set up the executor instance and initial state."""
        self.executor = ManimScriptExecutor()
        self.valid_state: GraphState = {
            "generated_output": "from manim import Scene\nclass TestScene(Scene): pass",
            "iteration": 1,
            "error_history": [],
            "evaluation_history": [],
            # Add other required keys from GraphState with default values if needed
            "input_text": "",
            "context_doc": "",
            "rubric": "",
            "input_metadata": None,
            "max_iterations": 5,
            "validation_error": None,
            "validated_artifact_path": None,
            "evaluation_feedback": None,
            "evaluation_passed": None,
        }
        # Clean up temp dir before/after tests if it exists
        if os.path.exists(MockAgentConfig.TEMP_SCRIPT_DIR):
            # Simple cleanup, might need shutil.rmtree if nested dirs are created
            for f in os.listdir(MockAgentConfig.TEMP_SCRIPT_DIR):
                os.remove(os.path.join(MockAgentConfig.TEMP_SCRIPT_DIR, f))
            os.rmdir(MockAgentConfig.TEMP_SCRIPT_DIR)

    def tearDown(self):
        if os.path.exists(MockAgentConfig.TEMP_SCRIPT_DIR):
            for f in os.listdir(MockAgentConfig.TEMP_SCRIPT_DIR):
                os.remove(os.path.join(MockAgentConfig.TEMP_SCRIPT_DIR, f))
            os.rmdir(MockAgentConfig.TEMP_SCRIPT_DIR)

    @patch("agents.manim_agent.script_executor.os.makedirs")
    @patch("agents.manim_agent.script_executor.uuid.uuid4")
    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.manim_agent.script_executor.subprocess.run")
    @patch("agents.manim_agent.script_executor.os.path.exists")
    @patch("agents.manim_agent.script_executor.os.remove")
    def test_execute_manim_script_success(
        self,
        mock_os_remove,
        mock_os_path_exists,
        mock_subprocess_run,
        mock_open_file,
        mock_uuid,
        mock_makedirs,
    ):
        """Tests successful script execution and video validation."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        temp_script_name = "manim_script_12345678-1234-5678-1234-567812345678.py"
        temp_script_path = os.path.join(MockAgentConfig.TEMP_SCRIPT_DIR, temp_script_name)
        expected_video_rel_path = os.path.join(
            MockAgentConfig.MANIM_VIDEO_OUTPUT_RELATIVE_PATH,
            MockAgentConfig.EXPECTED_VIDEO_FILENAME,
        )
        expected_video_full_path = os.path.join(MockBaseConfig.BASE_DIR, expected_video_rel_path)

        # Mock subprocess result for success
        mock_proc_result = MagicMock()
        mock_proc_result.returncode = 0
        mock_proc_result.stdout = "Manim execution log"
        mock_proc_result.stderr = ""
        mock_subprocess_run.return_value = mock_proc_result

        # Mock os.path.exists to return True for the video file and the temp script
        mock_os_path_exists.side_effect = (
            lambda path: path == expected_video_full_path or path == temp_script_path
        )

        result_state = self.executor.execute_manim_script(self.valid_state)

        mock_makedirs.assert_called_once_with(MockAgentConfig.TEMP_SCRIPT_DIR, exist_ok=True)
        mock_open_file.assert_called_once_with(temp_script_path, "w", encoding="utf-8")
        mock_open_file().write.assert_called_once_with(self.valid_state["generated_output"])
        mock_subprocess_run.assert_called_once()
        # Check command args
        expected_command = [
            "python",
            "-m",
            "manim",
            MockAgentConfig.MANIM_QUALITY_FLAG,
            temp_script_path,
            MockAgentConfig.GENERATED_SCENE_NAME,
            "--output_dir",
            ".",
        ]
        call_args, call_kwargs = mock_subprocess_run.call_args
        self.assertEqual(call_args[0], expected_command)
        self.assertEqual(call_kwargs.get("cwd"), MockBaseConfig.BASE_DIR)

        # Check that os.path.exists was called for the video file
        mock_os_path_exists.assert_any_call(expected_video_full_path)
        self.assertEqual(result_state["validated_artifact_path"], expected_video_rel_path)
        self.assertIsNone(result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 0)
        mock_os_remove.assert_called_once_with(temp_script_path)

    def test_execute_manim_script_no_code(self):
        """Tests execution attempt when no code is in the state."""
        no_code_state = self.valid_state.copy()
        no_code_state["generated_output"] = None

        result_state = self.executor.execute_manim_script(no_code_state)

        self.assertIsNone(result_state["validated_artifact_path"])
        self.assertIsNotNone(result_state["validation_error"])
        self.assertIn("No Manim code provided", result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn("Iter 1: No Manim code provided", result_state["error_history"][0])

    @patch("agents.manim_agent.script_executor.os.makedirs")
    @patch("agents.manim_agent.script_executor.uuid.uuid4")
    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.manim_agent.script_executor.subprocess.run")
    @patch("agents.manim_agent.script_executor.os.path.exists")
    @patch("agents.manim_agent.script_executor.os.remove")
    def test_execute_manim_script_execution_error(
        self,
        mock_os_remove,
        mock_os_path_exists,
        mock_subprocess_run,
        mock_open_file,
        mock_uuid,
        mock_makedirs,
    ):
        """Tests handling when Manim execution fails (non-zero return code)."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        temp_script_name = "manim_script_12345678-1234-5678-1234-567812345678.py"
        temp_script_path = os.path.join(MockAgentConfig.TEMP_SCRIPT_DIR, temp_script_name)

        # Mock subprocess result for failure
        mock_proc_result = MagicMock()
        mock_proc_result.returncode = 1
        mock_proc_result.stdout = ""
        mock_proc_result.stderr = "Manim error details here"
        mock_subprocess_run.return_value = mock_proc_result

        # Mock os.path.exists for the temp script cleanup
        mock_os_path_exists.return_value = True  # Assume temp script exists for cleanup

        result_state = self.executor.execute_manim_script(self.valid_state)

        mock_subprocess_run.assert_called_once()
        self.assertIsNone(result_state["validated_artifact_path"])
        self.assertIsNotNone(result_state["validation_error"])
        self.assertIn("Manim execution failed", result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn("Iter 1: Manim Execution Error", result_state["error_history"][0])
        self.assertIn("Manim error details here", result_state["error_history"][0])
        mock_os_remove.assert_called_once_with(temp_script_path)

    @patch("agents.manim_agent.script_executor.os.makedirs")
    @patch("agents.manim_agent.script_executor.uuid.uuid4")
    @patch("builtins.open", new_callable=mock_open)
    @patch("agents.manim_agent.script_executor.subprocess.run")
    @patch("agents.manim_agent.script_executor.os.path.exists")
    @patch("agents.manim_agent.script_executor.os.remove")
    def test_execute_manim_script_video_not_found(
        self,
        mock_os_remove,
        mock_os_path_exists,
        mock_subprocess_run,
        mock_open_file,
        mock_uuid,
        mock_makedirs,
    ):
        """Tests handling when Manim succeeds but the video file is not found."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        temp_script_name = "manim_script_12345678-1234-5678-1234-567812345678.py"
        temp_script_path = os.path.join(MockAgentConfig.TEMP_SCRIPT_DIR, temp_script_name)
        expected_video_rel_path = os.path.join(
            MockAgentConfig.MANIM_VIDEO_OUTPUT_RELATIVE_PATH,
            MockAgentConfig.EXPECTED_VIDEO_FILENAME,
        )
        expected_video_full_path = os.path.join(MockBaseConfig.BASE_DIR, expected_video_rel_path)

        # Mock subprocess result for success
        mock_proc_result = MagicMock()
        mock_proc_result.returncode = 0
        mock_proc_result.stdout = "Manim execution log"
        mock_proc_result.stderr = ""
        mock_subprocess_run.return_value = mock_proc_result

        # Mock os.path.exists to return False for the video file, True for temp script
        mock_os_path_exists.side_effect = lambda path: path == temp_script_path

        result_state = self.executor.execute_manim_script(self.valid_state)

        mock_subprocess_run.assert_called_once()
        # Check that os.path.exists was called for the video file
        mock_os_path_exists.assert_any_call(expected_video_full_path)
        self.assertIsNone(result_state["validated_artifact_path"])
        self.assertIsNotNone(result_state["validation_error"])
        self.assertIn("expected video not found", result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn(
            "Iter 1: Manim reported success, but expected video not found",
            result_state["error_history"][0],
        )
        mock_os_remove.assert_called_once_with(temp_script_path)


if __name__ == "__main__":
    unittest.main()
