"""
Command Line Interface for AI Data Synthesizer
Provides interactive CLI mode for testing and development
"""

import sys
import os
import uuid

# UTF-8 Fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logger.logger import get_logger
from config.settings import get_settings
from core.data_synthesizer import DataSynthesizer
from core.input_processor import InputProcessor
from input.cli_input_collector import collect_generation_input


def main():
    logger = get_logger(__name__)
    settings = get_settings()

    # Configure AWS credentials (Bedrock access)
    try:
        settings.configure_aws_credentials()
    except Exception:
        pass

    # ---------------- CLI BANNER ----------------
    print("\n" + "=" * 60)
    print(" AI DATA SYNTHESIZER - Interactive Mode")
    print("=" * 60)
    print("Type 'generate' to create dataset")
    print("Type 'exit' to stop")
    print("=" * 60)

    synthesizer = DataSynthesizer()
    input_processor = InputProcessor()

    # ---------------- MAIN LOOP ----------------
    while True:
        try:
            command = input("\n> ").strip()

            if not command:
                continue

            if command.lower() in ["exit", "quit", "q"]:
                print(" Goodbye!")
                break

            if command.lower() == "generate":
                raw_request = collect_generation_input()
                request = input_processor.build_request(raw_request)

                # STEP 1: Show preview, ask for confirmation
                confirmed = show_preview(request)
                if not confirmed:
                    print("  ↩ Generation cancelled.")
                    continue

                # STEP 2: Call LLM and generate
                print("\n  Generating... (calling Bedrock LLM)")
                output = synthesizer.generate(request)
                display_results(output)
            else:
                print("Unknown command. Type 'generate' or 'exit'.\"")

        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            logger.error(f"CLI error: {e}", exc_info=True)
            print(f"Error: {e}")


def show_preview(request: dict) -> bool:
    """
    Display a summary of what will be generated.
    Ask user to confirm before calling the LLM.
    Returns True if confirmed, False to cancel.
    """
    print("\n" + "─" * 60)
    print("  GENERATION PREVIEW — Please review before confirming")
    print("─" * 60)
    print(f"  Dataset Name  : {request.get('dataset_name')}")
    print(f"  Rows          : {request.get('rows')}")
    print(f"  Format        : {request.get('format')}")

    if request.get("description"):
        print(f"  Description   : {request['description']}")
    if request.get("ai_criteria"):
        print(f"  AI Criteria   : {request['ai_criteria']}")
    if request.get("target_location"):
        print(f"  Save To       : {request['target_location']}")

    schema = request.get("schema")
    if schema:
        print(f"\n  {'Column':<20} {'Type':<12} {'Nullable':<10} {'Pattern'}")
        print("  " + "─" * 55)
        for col in schema:
            name     = col.get("name", "")
            col_type = col.get("type", "string")
            nullable = "yes" if col.get("nullable") else "no"
            pattern  = col.get("pattern", "") or ""
            print(f"  {name:<20} {col_type:<12} {nullable:<10} {pattern}")
        print("  " + "─" * 55)
    else:
        print("\n  Schema        : Not defined — AI will decide columns")

    print("\n" + "─" * 60)

    confirm = input("  Confirm and generate? (y/n) [y]: ").strip().lower()
    return confirm in ("", "y", "yes")


def display_results(output):
    print("\n" + "=" * 60)

    if isinstance(output, dict):
        print(" Generation Summary:")
        for key, value in output.items():
            print(f"  {key}: {value}")
    else:
        print(" Response:")
        print(output)

    print("=" * 60)


if __name__ == "__main__":
    main()