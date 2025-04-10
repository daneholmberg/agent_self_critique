import unittest
from unittest.mock import patch, MagicMock

from agents.manim_agent.code_generator import ManimCodeGenerator
from core.graph_state import GraphState
from langchain_google_genai import ChatGoogleGenerativeAI


# Mock config values needed for tests
class MockAgentConfig:
    GENERATED_SCENE_NAME = "TestScene"


@patch("agents.manim_agent.code_generator.agent_cfg", MockAgentConfig)
class TestManimCodeGenerator(unittest.TestCase):

    def setUp(self):
        """Set up a mock LLM client and the generator instance."""
        self.mock_llm_client = MagicMock(spec=ChatGoogleGenerativeAI)
        self.mock_llm_client.model = "mock-gemini-model"
        self.generator = ManimCodeGenerator(llm_text_client=self.mock_llm_client)
        self.initial_state: GraphState = {
            "input_text": "Draw a circle.",
            "context_doc": "Manim context here.",
            "rubric": "Rubric details here.",
            "iteration": 0,
            "error_history": [],
            "evaluation_history": [],
            # Add other required keys from GraphState with default values if needed
            "input_metadata": None,
            "max_iterations": 5,
            "generated_output": None,
            "validation_error": None,
            "validated_artifact_path": None,
            "evaluation_feedback": None,
            "evaluation_passed": None,
        }

    def test_extract_python_code_valid(self):
        """Tests extracting code from standard markdown block."""
        text = "Some text\n```python\nprint('hello')\n```\nMore text"
        extracted = self.generator._extract_python_code(text)
        self.assertEqual(extracted, "print('hello')")

    def test_extract_python_code_no_block(self):
        """Tests extraction when no markdown block is present."""
        text = "Just plain text without code block."
        extracted = self.generator._extract_python_code(text)
        self.assertIsNone(extracted)

    def test_extract_python_code_fallback_heuristic(self):
        """Tests the fallback heuristic for code without fences."""
        text = "import os\ndef my_func():\n    pass"  # Looks like code
        extracted = self.generator._extract_python_code(text)
        self.assertEqual(extracted, text)

    def test_build_generation_prompt_initial(self):
        """Tests the initial prompt structure."""
        prompt = self.generator._build_generation_prompt(self.initial_state, 1)
        self.assertIn("You are a Manim expert.", prompt)
        self.assertIn("Draw a circle.", prompt)
        self.assertIn("Manim context here.", prompt)
        self.assertNotIn("Previous Attempt Feedback", prompt)
        self.assertIn(f"scene named '{MockAgentConfig.GENERATED_SCENE_NAME}'", prompt)

    def test_build_generation_prompt_retry_with_error(self):
        """Tests prompt structure on retry with validation error."""
        retry_state = self.initial_state.copy()
        retry_state["validation_error"] = "Syntax Error on line 5"
        prompt = self.generator._build_generation_prompt(retry_state, 2)
        self.assertIn("Previous Attempt Feedback", prompt)
        self.assertIn("failed validation/execution", prompt)
        self.assertIn("Syntax Error on line 5", prompt)
        self.assertNotIn("evaluation feedback", prompt)

    def test_build_generation_prompt_retry_with_eval(self):
        """Tests prompt structure on retry with evaluation feedback."""
        retry_state = self.initial_state.copy()
        retry_state["evaluation_feedback"] = "Video is too fast"
        prompt = self.generator._build_generation_prompt(retry_state, 2)
        self.assertIn("Previous Attempt Feedback", prompt)
        self.assertNotIn("failed validation/execution", prompt)
        self.assertIn("evaluation feedback", prompt)
        self.assertIn("Video is too fast", prompt)

    @patch("agents.manim_agent.code_generator.compile")
    def test_generate_manim_code_success(self, mock_compile):
        """Tests successful code generation and basic validation."""
        mock_response = MagicMock()
        mock_response.content = "```python\nfrom manim import Scene, Circle\n\nclass TestScene(Scene):\n    def construct(self):\n        c = Circle()\n        self.play(Create(c))\n```"
        self.mock_llm_client.invoke.return_value = mock_response
        mock_compile.return_value = None  # Simulate successful compile

        result_state = self.generator.generate_manim_code(self.initial_state)

        self.mock_llm_client.invoke.assert_called_once()
        mock_compile.assert_called_once()
        self.assertEqual(result_state["iteration"], 1)
        self.assertIsNotNone(result_state["generated_output"])
        self.assertIn("class TestScene(Scene):", result_state["generated_output"])
        self.assertIsNone(result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 0)

    def test_generate_manim_code_llm_error(self):
        """Tests handling of LLM API errors during generation."""
        self.mock_llm_client.invoke.side_effect = Exception("API limit reached")

        result_state = self.generator.generate_manim_code(self.initial_state)

        self.mock_llm_client.invoke.assert_called_once()
        self.assertIsNone(result_state["generated_output"])
        self.assertIsNotNone(result_state["validation_error"])
        self.assertIn("Gemini API Error", result_state["validation_error"])
        self.assertIn("API limit reached", result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn("Iter 1: Gemini API Error", result_state["error_history"][0])

    def test_generate_manim_code_parsing_error(self):
        """Tests handling when LLM response cannot be parsed."""
        mock_response = MagicMock()
        mock_response.content = "Sorry, I cannot generate the code."
        self.mock_llm_client.invoke.return_value = mock_response

        result_state = self.generator.generate_manim_code(self.initial_state)

        self.mock_llm_client.invoke.assert_called_once()
        self.assertIsNone(result_state["generated_output"])
        self.assertIsNotNone(result_state["validation_error"])
        self.assertIn("Failed to parse Python code block", result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn("Iter 1: Failed to parse", result_state["error_history"][0])

    @patch("agents.manim_agent.code_generator.compile", side_effect=SyntaxError("invalid syntax"))
    def test_generate_manim_code_syntax_error(self, mock_compile):
        """Tests handling when generated code has a syntax error."""
        mock_response = MagicMock()
        # Code missing closing parenthesis
        mock_response.content = "```python\nfrom manim import Scene, Circle\n\nclass TestScene(Scene):\n    def construct(self):\n        c = Circle(\n        self.play(Create(c))\n```"
        self.mock_llm_client.invoke.return_value = mock_response

        result_state = self.generator.generate_manim_code(self.initial_state)

        self.mock_llm_client.invoke.assert_called_once()
        mock_compile.assert_called_once()
        # Output is generated but validation fails
        self.assertIsNotNone(result_state["generated_output"])
        self.assertIsNotNone(result_state["validation_error"])
        self.assertIn("SyntaxError", result_state["validation_error"])
        self.assertEqual(len(result_state["error_history"]), 1)
        self.assertIn("Iter 1: Generated code has SyntaxError", result_state["error_history"][0])


if __name__ == "__main__":
    unittest.main()
