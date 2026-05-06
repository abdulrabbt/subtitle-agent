"""CLI entry point for the Subtitle Translation Agent.

Usage:
    python main.py input/TheMatrix.srt output/TheMatrix.ar.srt
    python main.py TheMatrix.srt TheMatrix.ar.srt --debug
"""

import sys
import logging
from pathlib import Path

from src.agent import run_translation

# Set up root logger — level will be adjusted by --debug flag
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
root_logger = logging.getLogger()


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input.srt> <output.srt> [--debug]")
        print("Example: python main.py input/TheMatrix.srt output/TheMatrix.ar.srt --debug")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    debug = "--debug" in sys.argv

    if debug:
        root_logger.setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.debug("DEBUG MODE enabled — verbose logging active")

    # Auto-create input/output directories if needed
    Path(input_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    try:
        run_translation(input_path, output_path)
    except KeyboardInterrupt:
        print("\nTranslation interrupted. Progress saved — re-run to resume.")
        sys.exit(1)
    except Exception as e:
        logging.error("Fatal error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()