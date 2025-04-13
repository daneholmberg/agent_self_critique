import logging
from typing import Dict, Any
from pathlib import Path
from langchain_core.language_models import BaseLanguageModel
from core.log_utils import log_run_details

# Configure logging
logger = logging.getLogger(__name__)


class FailureSummarizer:
    """
    Component responsible for summarizing individual failure reasons using an LLM.
    """

    def __init__(self, llm: BaseLanguageModel, summarization_prompt_template: str):
        """
        Initializes the FailureSummarizer.

        Args:
            llm: The language model instance to use for summarization.
            summarization_prompt_template: A format string template for the summarization prompt.
                                          It should expect a 'failure_detail' key.
        """
        if not llm:
            raise ValueError("LLM instance is required for FailureSummarizer.")
        self.llm = llm
        self.summarization_prompt_template = summarization_prompt_template
        logger.info(f"FailureSummarizer initialized with LLM: {llm.__class__.__name__}")

    async def summarize(
        self, failure_detail: str, run_output_dir: Path | str, attempt_number_failed: int
    ) -> str:
        """
        Generates a concise summary for a single failure detail using the configured LLM.

        Args:
            failure_detail: The raw error message or rubric failure reason.
            run_output_dir: The directory for the current run's output logs.
            attempt_number_failed: The attempt number (1-based) that *failed*.

        Returns:
            A concise, token-optimized summary of the failure.
        """
        node_name = "FailureSummarizer"
        # Log Entry
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=attempt_number_failed,
            node_name=node_name,
            log_category="Node Entry",
            content=f"Starting summary for failure of attempt {attempt_number_failed}: {failure_detail[:100]}...",
        )

        prompt = self.summarization_prompt_template.format(failure_detail=failure_detail)

        # Log Prompt
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=attempt_number_failed,
            node_name=node_name,
            log_category="LLM Prompt",
            content=prompt,
        )

        summary = "[Summarization Failed]"
        try:
            response = await self.llm.ainvoke(prompt, config={"configurable": {"temperature": 0.5}})
            raw_summary = (
                response.content.strip() if hasattr(response, "content") else str(response).strip()
            )

            if not raw_summary:
                logger.warning(
                    f"LLM generated an empty summary for failure of attempt {attempt_number_failed}."
                )
                summary = f"[Summarization Failed - Empty] Original: {failure_detail[:150]}..."
                log_run_details(
                    run_output_dir=run_output_dir,
                    attempt_number=attempt_number_failed,
                    node_name=node_name,
                    log_category="Warning",
                    content="LLM returned empty summary.",
                    is_error=True,
                )
            else:
                summary = raw_summary
                logger.info(f"Generated failure summary: {summary}")
                # Log only the successful summary here
                log_run_details(
                    run_output_dir=run_output_dir,
                    attempt_number=attempt_number_failed,
                    node_name=node_name,
                    log_category="LLM Response (Summary)",
                    content=summary,
                )

        except Exception as e:
            error_message = f"Error during failure summarization LLM call: {e}"
            logger.error(error_message, exc_info=True)
            summary = f"[Summarization Error: {e}] Original: {failure_detail[:150]}..."
            # Log LLM Error
            log_run_details(
                run_output_dir=run_output_dir,
                attempt_number=attempt_number_failed,
                node_name=node_name,
                log_category="LLM Error",
                content=error_message,
                is_error=True,
            )

        # Log Completion (includes the final summary or error state)
        log_run_details(
            run_output_dir=run_output_dir,
            attempt_number=attempt_number_failed,
            node_name=node_name,
            log_category="Node Completion",
            content=f"Finished summarization for attempt {attempt_number_failed}. Result: {summary}",
            is_error=("[Summarization Failed" in summary or "[Summarization Error" in summary),
        )
        return summary


# Example Usage (Illustrative - Actual instantiation happens in runner.py)
# if __name__ == '__main__':
#     # This block is for testing/demonstration purposes only
#     from langchain_google_genai import ChatGoogleGenerativeAI # Example LLM
#     from agents.manim_agent.config import get_gemini_api_key, TEXT_GENERATION_MODEL
#
#     # Example prompt template
#     DEFAULT_FAILURE_SUMMARY_PROMPT = (
#         "Given the following failure reason encountered during Manim video generation, "
#         "provide a very concise, token-optimized summary (less than 30 words) focusing on the core issue. "
#         "Do not include apologies or suggestions for fixes, just state the problem clearly.\n\n"
#         "Failure Reason:\n{failure_detail}\n\n"
#         "Concise Summary:"
#     )
#
#     # Example LLM setup (replace with actual factory/config access)
#     api_key = get_gemini_api_key()
#     llm = ChatGoogleGenerativeAI(model=TEXT_GENERATION_MODEL, google_api_key=api_key)
#
#     summarizer = FailureSummarizer(llm=llm, summarization_prompt_template=DEFAULT_FAILURE_SUMMARY_PROMPT)
#
#     # Example failure detail
#     example_error = "Traceback (most recent call last):\n...\nmanim.mobject.geometry.Dot.set_color(self, color=['#FF0000'])\n...\nValueError: Could not interpret color '#FF0000'"
#
#     async def run_summary():
#         summary = await summarizer.summarize(example_error)
#         print(f"\nSummary:\n{summary}")
#
#     import asyncio
#     asyncio.run(run_summary())
