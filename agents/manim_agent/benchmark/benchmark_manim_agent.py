import asyncio
import datetime
import json
from pathlib import Path
import sys
import logging  # Import logging
import questionary  # Add questionary import

# Add project root to sys.path to allow imports from core/ and agents/
# This might need adjustment based on your project structure
# Assuming this script is in agents/manim_agent/benchmark/
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the manim agent runner's execute function
from agents.manim_agent.runner import execute as run_manim_agent

# --- Configuration ---

# Define the 3 prompts with scene names
PROMPTS = [
    (
        "MMOSNetworkGrowth",
        """Goal: To flawlessly visualize the unprecedented scale increase and the crucial role of MMOS in facilitating massive collaboration, making the script's meaning instantly graspable through the network metaphor.
Refined Scene Description:
Initial State (Emphasize Sparseness): Start with a starkly minimal network graph on the left: very few nodes (visually representing maybe <100k players), sparsely connected with faint lines. Label clearly: "Previous Citizen Science Games." The surrounding area is dark and empty, emphasizing limitations.
Introduce Challenge (Visually Urgent): A stylized 'Professor' icon appears, looking towards the vast, empty right side. Text boldly fades in above this space: "Needed: Millions of Players". The emptiness feels vast and challenging.
The Connection (Clear Catalyst): An animated, glowing line pulses out from the 'Professor' icon, decisively connecting to a central 'MMOS' logo that appears with a subtle pulse of light/energy. This connection feels like the activation key.
Explosive, Dense Growth (Overwhelm with Clarity): Immediately upon connection, the MMOS logo becomes a vibrant source. Thousands of tiny, bright nodes (initially like particles, signifying sheer number) explode outwards, rapidly populating the empty grid. These nodes instantly begin forming connections with bright, animated lines, creating a massive, incredibly dense, and active network graph. The growth should feel almost overwhelming but remain visually clear (e.g., nodes/lines maintain distinctness). Crucially, the sheer density and speed of this growth directly visualizes "millions" and "unprecedented scale" far exceeding the initial sparse network.
Final View (Extreme Contrast): Hold briefly on the final composition: the tiny, sparse initial network on the left, the Professor icon, the glowing MMOS logo acting as the central hub, and the enormous, hyper-dense, vibrant network filling the right side. The visual contrast in scale and density is extreme and undeniable, perfectly mirroring the script's narrative of scale difference and MMOS's role as the facilitator of this massive collaborative network.
The scene is 23 seconds""",
    ),
    (
        "BorderlandsPuzzleMMOS",
        """Goal: To seamlessly integrate the concepts of the massive task (puzzles) and the massive player base needed, showing MMOS as the essential connector, ensuring effortless comprehension even with potentially complex visuals.
Refined Scene Description:
Initial State (Clear Baseline): Display a small, contained graph/network diagram (representing a manageable scientific task) on the left, clearly labeled "Previous Citizen Science Games". Around it, show a few hundred small, faint dots (players), sparsely interacting with it.
Transition & Scale Shift (Task Focus First): Animate the left elements shrinking/moving aside. Simultaneously, a vastly larger, visually distinct (e.g., geometric grid, complex structure) representation of the task fades in, dominating the screen. Label boldly: "Borderlands Science: Millions of Puzzles". The sheer scale of the problem is established first.
Introduce Need & Connector (Sequential Clarity): A single, distinct 'Professor' icon appears near the massive puzzle structure, perhaps looking up at it to emphasize the challenge. Text briefly appears: "Millions of Players Required". Then, a central 'MMOS' hub icon fades in strategically (e.g., below the puzzle grid, between Professor and empty space).
The MMOS Connection & Player Influx (Visually Distinct & Overwhelming): A bright, clear line animates from the 'Professor' to the 'MMOS' hub. The moment it connects, the MMOS hub pulses brightly. Instantly, from or near the MMOS hub, an overwhelming swarm of millions of bright, visually distinct particles (e.g., glowing dots, different color/style than the puzzle structure) floods the screen, surrounding the puzzle structure. Crucially, numerous bright lines shoot specifically from the MMOS hub into this swarm, clearly showing MMOS connecting to the players.
Final View (Synergy & Purpose): Hold on the scene: The Professor connected to the MMOS hub, the MMOS hub acting as the source/connector for the massive, distinct player particle swarm, which now envelops the huge puzzle structure. (Optional: Faint lines could subtly animate from the particle swarm towards the puzzle structure, showing the players beginning to engage, reinforcing the facilitated collaboration). The visual hierarchy clearly separates task, facilitator, and the massive player base, making the entire relationship and scale instantly understandable alongside the script.
The scene is 23 seconds""",
    ),
    (
        "puzzleMountain",
        """Goal: To use the potent "Puzzle Mountain" metaphor while ensuring crystal-clear synergy with every part of the script, especially the scale comparison and the active role of MMOS as the essential facilitator.
Refined Scene Description:
Establish Baseline (Crucial Contrast): Start clearly on the left: A small, manageable "hill" made of maybe ~50 distinct puzzle piece icons. Below it, a small cluster of ~30 player icons interacting with the hill. Label clearly: "Previous Citizen Science Games". The visual feels contained and achievable.
Dramatic Scale Shift (The Unprecedented): Animate a rapid zoom-out or a stark transition. The small hill and players shrink/slide away. A massive, visually imposing "Mountain" composed of countless, densely packed, smaller puzzle piece icons dominates the screen. It should look genuinely daunting. Text appears prominently near it: "Borderlands Science: Millions of Puzzles".
Explicit Need & The Professor: The lone 'Professor' icon appears, small against the mountain's base, perhaps looking up. Immediately, text appears near the Professor, directly linking the mountain to the requirement: "Required: Millions of Players". The inadequacy of the "previous games" scale is now visually obvious by comparison.
MMOS as the Active Gateway: The 'Professor' icon seems to 'activate' something. A bright line shoots from the Professor to connect decisively with the 'MMOS' logo, which fades in centrally (perhaps positioned like a bridge or portal below the mountain). The MMOS logo pulses with energy upon connection.
Facilitated Player Access (Clear Source & Flow): Instead of just a cloud appearing near MMOS, bright streams of light or distinct particles (representing players) flow directly out of the activated MMOS logo. This massive stream flows upwards/outwards, forming a huge, dynamic cloud/swarm that surrounds the base and lower slopes of the Puzzle Mountain. This visually confirms MMOS as the source or gateway providing access to the required millions.
Final View (Clear Relationships): Hold briefly: The tiny remnant of the "previous" scale (optional, could be faded), the imposing Puzzle Mountain (the challenge), the small Professor (the need), the glowing MMOS logo acting as the explicit gateway, and the massive swarm of players flowing from MMOS to address the mountain. The visual story perfectly mirrors the script: unprecedented scale requires millions, Professor connects to MMOS, MMOS facilitates access to the players.""",
    ),
    (
        "simple",
        """Create a simple scene with a single object. But be a little creative. Make it interesting.""",
    ),
    (
        "superSimple",
        "Create a simple scene with a single circle animated in",
    ),
]

# Define the base output directory for this benchmark run
BENCHMARK_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Place benchmark_outputs at the project root
BENCHMARK_BASE_DIR = project_root / "agents/manim_agent/benchmark/benchmark_outputs"
BENCHMARK_OUTPUT_DIR = BENCHMARK_BASE_DIR / f"benchmark_{BENCHMARK_TIMESTAMP}"
BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Agent configuration defaults will be used (rubric path, max_attempts)
SAVE_GENERATED_CODE = True  # Or False, depending on preference

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Create a logger instance


# --- Benchmark Execution ---


async def run_single_benchmark(
    prompt_name: str, prompt_text: str, prompt_index: int, base_output_dir: Path
):
    """Runs the Manim agent for a single prompt and manages output."""
    logger.info(f"--- Starting Benchmark for Prompt {prompt_index + 1} ({prompt_name}) ---")
    logger.info(f"Prompt Text: {prompt_text[:100]}...")  # Log truncated text

    # Create a dedicated output directory for this specific prompt run
    # Use the prompt name for a more descriptive directory
    prompt_output_dir_name = f"prompt_{prompt_index + 1}_{prompt_name}"
    prompt_output_dir = base_output_dir / prompt_output_dir_name
    # The agent runner will create this directory if needed, via run_output_dir_override

    try:
        # Context and Rubric paths will use defaults within the agent runner
        result = await run_manim_agent(
            script_segment=prompt_text,  # Pass the text part of the tuple
            # context_path=str(CONTEXT_PATH.resolve()), # Use agent default
            # rubric_path=str(RUBRIC_PATH.resolve()),   # Use agent default
            # max_attempts=MAX_ATTEMPTS,                # Use agent default
            scene_name=prompt_name,
            save_generated_code=SAVE_GENERATED_CODE,
            run_output_dir_override=str(
                prompt_output_dir.resolve()
            ),  # *** Crucial: Override output dir ***
            # Add other necessary parameters if needed (e.g., llm_config)
        )
        logger.info(f"--- Finished Benchmark for Prompt {prompt_index + 1} ({prompt_name}) ---")
        return {
            "prompt_index": prompt_index,
            "prompt_name": prompt_name,
            "prompt_text": prompt_text,
            "output_dir": str(prompt_output_dir),
            "result": result,
            "status": "success",
        }

    except Exception as e:
        logger.error(
            f"--- Error during Benchmark for Prompt {prompt_index + 1} ({prompt_name}) ---",
            exc_info=True,
        )
        # Ensure the output dir exists for logging even on error, if possible
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "prompt_index": prompt_index,
            "prompt_name": prompt_name,
            "prompt_text": prompt_text,
            "output_dir": str(prompt_output_dir),
            "error": str(e),
            "status": "error",
        }


async def main():
    """Runs the benchmark, allowing prompt selection."""
    all_results = []

    # Generate choices using the prompt name and truncated text
    prompt_choices = [
        f"Prompt {i+1} ({name}): {text[:60]}..." for i, (name, text) in enumerate(PROMPTS)
    ]
    all_option = "Run All Prompts"
    choices = prompt_choices + [all_option]

    selected_prompt_text = await questionary.select(
        "Which benchmark prompt would you like to run?",
        choices=choices,
    ).ask_async()  # Use ask_async for await

    if not selected_prompt_text:
        logger.info("No prompt selected. Exiting.")
        return

    if selected_prompt_text == all_option:
        logger.info("Running all prompts...")
        for i, (prompt_name, prompt_text) in enumerate(PROMPTS):
            run_result = await run_single_benchmark(
                prompt_name, prompt_text, i, BENCHMARK_OUTPUT_DIR
            )
            all_results.append(run_result)
    else:
        # Find the index of the selected prompt
        try:
            selected_index = choices.index(selected_prompt_text)
            if selected_index < len(PROMPTS):  # Ensure it's not "Run All" index
                selected_name, selected_text = PROMPTS[selected_index]
                logger.info(f"Running selected prompt: {selected_prompt_text}")
                run_result = await run_single_benchmark(
                    selected_name, selected_text, selected_index, BENCHMARK_OUTPUT_DIR
                )
                all_results.append(run_result)
            else:
                logger.error(
                    "Internal error: Selected choice index out of range."
                )  # Should not happen
                return
        except ValueError:
            logger.error(
                f"Error finding selected prompt: {selected_prompt_text}"
            )  # Should not happen
            return

    # --- Result Summary ---
    # Only save summary if any runs were attempted
    if all_results:
        summary_file = BENCHMARK_OUTPUT_DIR / "benchmark_summary.json"
        # Sort results by prompt index for consistency
        all_results.sort(key=lambda x: x["prompt_index"])
        with open(summary_file, "w") as f:
            # Attempt to serialize results, handling potential non-serializable objects
            try:
                json.dump(all_results, f, indent=4, default=lambda o: "<not serializable>")
            except TypeError as e:
                logger.error(f"Could not serialize all benchmark results to JSON: {e}")
                f.write(f"Error serializing results: {e}")
                # Optionally write a simpler representation
                for res in all_results:
                    f.write(f"Prompt {res['prompt_index']+1} Status: {res['status']}")

        logger.info(f"--- Benchmark Complete ---")
        logger.info(f"Results summary saved to: {summary_file}")
        for res in all_results:
            status_indicator = "✅ Success" if res["status"] == "success" else "❌ Error"
            logger.info(
                f"Prompt {res['prompt_index'] + 1} ({res.get('prompt_name', 'N/A')}): {status_indicator} - Output: {res.get('output_dir', 'N/A')}"
            )
            if res["status"] == "error":
                logger.info(f"  Error: {res.get('error')}")
    else:
        logger.info("No benchmark prompts were run.")


if __name__ == "__main__":
    # Setup asyncio event loop
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" in str(e):
            logger.warning(
                "Event loop already running. Attempting to attach to existing loop (may not work in all environments)."
            )
            # This is a basic attempt; might need more robust handling depending on environment
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise  # Re-raise other RuntimeError exceptions
