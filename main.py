"""CLI entry point for the Subtitle Translation Agent.

Usage:
    python main.py input.srt output.srt
    python main.py input.srt output.srt --source-lang en --target-lang fr
    python main.py input.srt output.srt --target-lang es --debug
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.agent import run_translation

load_dotenv()

# Set up root logger — level will be adjusted by --debug flag
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
root_logger = logging.getLogger()


def _parse_args():
    """Parse CLI arguments manually for minimal overhead."""
    args = {
        "input": None,
        "output": None,
        "source_lang": os.getenv("SOURCE_LANG", "en"),
        "target_lang": os.getenv("TARGET_LANG", "ar"),
        "debug": False,
    }

    positional: list[str] = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--debug":
            args["debug"] = True
        elif arg == "--source-lang":
            i += 1
            if i < len(sys.argv):
                args["source_lang"] = sys.argv[i]
            else:
                print("Error: --source-lang requires a value", file=sys.stderr)
                sys.exit(1)
        elif arg == "--target-lang":
            i += 1
            if i < len(sys.argv):
                args["target_lang"] = sys.argv[i]
            else:
                print("Error: --target-lang requires a value", file=sys.stderr)
                sys.exit(1)
        elif not arg.startswith("-"):
            positional.append(arg)
        else:
            print(f"Error: Unknown option: {arg}", file=sys.stderr)
            sys.exit(1)
        i += 1

    if len(positional) >= 2:
        args["input"] = positional[0]
        args["output"] = positional[1]

    return args


def main():
    args = _parse_args()

    if not args["input"] or not args["output"]:
        print("Usage: python main.py <input.srt> <output.srt> [--source-lang CODE] [--target-lang CODE] [--debug]")
        print("Example: python main.py input.srt output.ar.srt --source-lang en --target-lang ar")
        print("Environment: Set SOURCE_LANG / TARGET_LANG in .env for defaults.")
        sys.exit(1)

    if args["debug"]:
        root_logger.setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.debug("DEBUG MODE enabled — verbose logging active")

    # Auto-create input/output directories if needed
    Path(args["input"]).parent.mkdir(parents=True, exist_ok=True)
    Path(args["output"]).parent.mkdir(parents=True, exist_ok=True)

    if not Path(args["input"]).exists():
        print(f"Error: Input file not found: {args['input']}")
        sys.exit(1)

    try:
        run_translation(
            input_path=args["input"],
            output_path=args["output"],
            source_lang=args["source_lang"],
            target_lang=args["target_lang"],
        )
    except KeyboardInterrupt:
        print("\nTranslation interrupted. Progress saved — re-run to resume.")
        sys.exit(1)
    except Exception as e:
        logging.error("Fatal error: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()