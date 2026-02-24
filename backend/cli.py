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
from core.dataset_generation_processor import DatasetGenerationProcessor
from interfaces.cli_input_collector import collect_generation_input


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

    session_id = str(uuid.uuid4())
    processor = DatasetGenerationProcessor(session_id=session_id)

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
                output = processor.process(raw_request)
                display_results(output)
            else:
                print("Unknown command. Type 'generate' or 'exit'.")

        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            logger.error(f"CLI error: {e}", exc_info=True)
            print(f"Error: {e}")


def display_results(output):
    print("\n" + "=" * 60)

    if isinstance(output, dict):
        print(" Generation Summary:")
        for key, value in output.items():
            print(f"{key}: {value}")
    else:
        print(" Response:")
        print(output)

    print("=" * 60)


if __name__ == "__main__":
    main()