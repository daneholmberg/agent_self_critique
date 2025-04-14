import asyncio
from typing import List, Optional, Dict, Any
import logging

from core.graph_state import GraphState

# from core.llm_clients import get_llm_client  # Removed old import
from core.llm.client_factory import create_llm_client  # Added new import
from agents.manim_agent import config as agent_cfg  # Added config import

logger = logging.getLogger(__name__)


class SceneIdeator:
    """Component responsible for generating multiple scene concepts based on input text."""

    # Removed temperature from __init__ as it's now fixed for this component
    def __init__(self, num_ideas: int = 3):
        """
        Initializes the SceneIdeator.

        Args:
            num_ideas: The number of distinct scene ideas to generate.
        """
        self.num_ideas = num_ideas
        # self.temperature = temperature # Removed temperature instance variable

        # Configure the specific model and temperature for ideation
        # self.llm_client = get_llm_client(temperature=self.temperature) # Removed old client creation
        self.llm_client = create_llm_client(
            provider="google",
            model_name=agent_cfg.TEXT_GENERATION_MODEL,  # Use the standard text gen model
            temperature=1.2,  # Set higher temperature for creative ideation
        )

        # Define the core prompt structure - this might need refinement
        self.system_prompt = (
            "You are a creative assistant specializing in visualizing concepts for Manim animations. "
            "Based on the user's input text and optional initial idea, generate a concise and compelling scene concept. "
            "Focus on the core visual elements and actions. Describe ONE distinct scene concept. DO NOT GENERATE CODE"
            # Add Manim-specific guidance if helpful, e.g., mentioning common objects or styles
            # "Consider using common Manim objects like Text, Square, Circle, Arrow, and transformations."
        )

    def _create_user_prompt(
        self, input_text: str, other_info: Optional[str], initial_idea: Optional[str]
    ) -> str:
        """Creates the user prompt for the LLM call."""
        prompt = ""
        if other_info:
            prompt += f"Other Relevant Information:\n```\n{other_info}\n```\n\n"
        prompt += f"Input Text:\n```\n{input_text}\n```\n\n"
        if initial_idea:
            prompt += f"Initial User Idea:\n```\n{initial_idea}\n```\n\n"
        prompt += """Generate a distinct Manim scene concept based on the above.
        The tools you have available besides all tools available in manim v0.19 are generating images with transparent backgrounds with gpt-4o 
        native multimodailty for better visuals than what you have with manim WHEN APPROPRIATE. We can also pull SVGs from online for better
        SVG creation. """
        return prompt

    async def _generate_single_idea(self, user_prompt: str) -> str:
        """Makes a single LLM call to generate one idea."""
        try:
            # Use asyncio.to_thread to run the synchronous invoke call in a separate thread
            response = await asyncio.to_thread(
                self.llm_client.invoke,
                [
                    ("system", self.system_prompt),
                    ("user", user_prompt),
                ],
            )
            # Assuming the response object has a 'content' attribute or similar
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating scene idea: {e}")
            return "Error: Could not generate idea."  # Provide a fallback

    async def ideate(
        self,
        input_text: str,
        other_info: Optional[str] = None,
        initial_idea: Optional[str] = None,
        num_ideas: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates multiple distinct scene ideas using parallel LLM calls.

        Args:
            input_text: The primary text description for the scene.
            other_info: Optional additional context or information.
            initial_idea: An optional initial concept provided by the user.
            num_ideas: Optional number of ideas to generate, overriding the instance default.

        Returns:
            A dictionary containing the list of generated ideas under the key 'generated_ideas'.
        """
        ideas_to_generate = num_ideas if num_ideas is not None and num_ideas > 0 else self.num_ideas
        logger.info(
            f"Generating {ideas_to_generate} scene ideas for input: '{input_text[:50]}...' with other info: '{str(other_info)[:50]}...' and initial idea: '{str(initial_idea)[:50]}...'"
        )

        user_prompt = self._create_user_prompt(input_text, other_info, initial_idea)

        # Generate ideas concurrently
        tasks = [self._generate_single_idea(user_prompt) for _ in range(ideas_to_generate)]
        generated_ideas = await asyncio.gather(*tasks)

        # Filter out any error messages if needed, though gather returns results in order
        valid_ideas = [idea for idea in generated_ideas if not idea.startswith("Error:")]

        logger.info(f"Generated {len(valid_ideas)} valid ideas.")

        return {"generated_ideas": valid_ideas}


# Example Usage (for testing)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     async def main():
#         ideator = SceneIdeator()
#         test_state = GraphState(
#             input_text="Explain the concept of photosynthesis using simple shapes.",
#             initial_idea="Maybe start with a sun and a plant?",
#             # Fill in other required fields for GraphState if necessary for LLM client
#             context_doc="", rubric="", max_iterations=1, iteration=0, error_history=[], evaluation_history=[], run_output_dir="", save_generated_code=False
#         )
#         result = await ideator.ideate(test_state)
#         print("Generated Ideas:")
#         for i, idea in enumerate(result.get('generated_ideas', [])):
#             print(f"{i+1}. {idea}")
#
#     asyncio.run(main())
