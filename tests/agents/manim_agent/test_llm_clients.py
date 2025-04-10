import unittest
from unittest.mock import patch, MagicMock

# Ensure the tested module can be imported. Adjust path if needed.
from agents.manim_agent.llm_clients import get_llm_clients
from langchain_google_genai import ChatGoogleGenerativeAI
from config import base_config


class TestLLMClients(unittest.TestCase):

    @patch("agents.manim_agent.llm_clients.base_cfg.get_gemini_api_key")
    @patch("agents.manim_agent.llm_clients.ChatGoogleGenerativeAI")
    def test_get_llm_clients_success(self, mock_chat_google, mock_get_key):
        """Tests successful initialization of both LLM clients."""
        mock_get_key.return_value = "DUMMY_API_KEY"
        # Mock the constructor to return mock instances
        mock_text_client = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_text_client.model = "gemini-text-model"  # Set a mock model attribute
        mock_eval_client = MagicMock(spec=ChatGoogleGenerativeAI)
        mock_eval_client.model = "gemini-eval-model"  # Set a mock model attribute
        mock_chat_google.side_effect = [mock_text_client, mock_eval_client]

        text_client, eval_client = get_llm_clients()

        self.assertIsInstance(text_client, MagicMock)  # Check if it's our mock
        self.assertIsInstance(eval_client, MagicMock)
        self.assertEqual(text_client.model, "gemini-text-model")
        self.assertEqual(eval_client.model, "gemini-eval-model")
        mock_get_key.assert_called_once()
        # Check if ChatGoogleGenerativeAI was called twice with expected args
        self.assertEqual(mock_chat_google.call_count, 2)
        # Could add more detailed checks on the args passed to ChatGoogleGenerativeAI constructor

    @patch("agents.manim_agent.llm_clients.base_cfg.get_gemini_api_key")
    @patch("agents.manim_agent.llm_clients.ChatGoogleGenerativeAI")
    def test_get_llm_clients_no_api_key(self, mock_chat_google, mock_get_key):
        """Tests that ValueError is raised if API key is not found."""
        mock_get_key.return_value = None

        with self.assertRaisesRegex(ValueError, "Gemini API key not found"):
            get_llm_clients()

        mock_get_key.assert_called_once()
        mock_chat_google.assert_not_called()  # Clients should not be initialized


if __name__ == "__main__":
    unittest.main()
