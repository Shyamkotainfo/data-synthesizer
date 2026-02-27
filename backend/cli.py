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
    print("  Modes: sdv (default) | llm (legacy Bedrock)")
    print("  SDV synthesizers: gaussian_copula | ctgan | tvae | copula_gan")
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

                # If we have sample rows but no schema yet, infer it now so the preview shows it
                if not request.get("schema") and request.get("sample_rows"):
                    from core.schema_resolver import SchemaResolver
                    request["schema"] = SchemaResolver.resolve(sample_rows=request["sample_rows"])
                    
                # STEP 1: Show preview of dataset and schema
                show_preview(request)

                # Ask about primary key if dealing with SDV
                if request.get("mode", "sdv") == "sdv" and request.get("schema"):
                    _confirm_or_change_primary_key(request["schema"])

                # STEP 2: Ask for final confirmation
                print("\n" + "─" * 60)
                confirm = input("  Confirm and generate? (y/n) [y]: ").strip().lower()
                if confirm not in ("", "y", "yes"):
                    print("  ↩ Generation cancelled.")
                    continue

                # STEP 2: Generate
                mode = request.get("mode", "sdv")
                if mode == "sdv":
                    print("\n  Generating... (SDV mode — training synthesizer on seed data)")
                else:
                    print("\n  Generating... (LLM mode — calling Bedrock)")
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


def _confirm_or_change_primary_key(schema: list):
    """
    Show all columns as a numbered list and let the user pick the primary key.
    """
    print("\n" + "─" * 60)
    print("  PRIMARY KEY / DEDUPLICATION COLUMN")
    print("  (Used to remove duplicate rows — does not change column format)")
    print("─" * 60)

    # Clear any existing PK flags first
    for col in schema:
        col["primary_key"] = False

    # Detect auto-suggested PK (uuid/id type)
    suggested_idx = None
    for i, col in enumerate(schema):
        if col.get("type", "").lower() in ("uuid", "id"):
            suggested_idx = i
            break

    # Display numbered list
    print("  Select the primary key column:\n")
    for i, col in enumerate(schema, 1):
        tag = "  ← suggested" if (i - 1) == suggested_idx else ""
        print(f"    {i:>2}. {col['name']:<25} ({col.get('type', 'string')}){tag}")

    print("     0. No primary key")
    print()

    default = str(suggested_idx + 1) if suggested_idx is not None else "0"
    while True:
        raw = input(f"  Enter number [{default}]: ").strip()
        if not raw:
            raw = default
        try:
            choice = int(raw)
        except ValueError:
            print("  ⚠ Please enter a valid number.")
            continue

        if choice == 0:
            print("  ℹ Proceeding without a primary key.")
            break
        elif 1 <= choice <= len(schema):
            pk_col = schema[choice - 1]
            pk_col["primary_key"] = True
            print(f"  ✅ Primary key set to [{pk_col['name']}]")
            break
        else:
            print(f"  ⚠ Enter a number between 0 and {len(schema)}.")



def show_preview(request: dict):
    """
    Display a summary of what will be generated.
    """
    print("\n" + "─" * 60)
    print("  GENERATION PREVIEW — Please review before confirming")
    print("─" * 60)
    print(f"  Dataset Name  : {request.get('dataset_name')}")
    print(f"  Rows          : {request.get('rows')}")
    print(f"  Format        : {request.get('format')}")
    mode = request.get('mode', 'sdv')
    synth = request.get('synthesizer', 'gaussian_copula')
    print(f"  Mode          : {mode}" + (f" ({synth})" if mode == 'sdv' else ""))

    if request.get("description"):
        print(f"  Description   : {request['description']}")
    if request.get("ai_criteria"):
        print(f"  AI Criteria   : {request['ai_criteria']}")
    if request.get("target_location"):
        print(f"  Save To       : {request['target_location']}")
    if request.get("sample_file"):
        print(f"  Sample Data   : {request['sample_file']} (Will be used to train SDV)")

    schema = request.get("schema")
    if schema:
        print(f"\n  {'Column':<25} {'Type':<12} {'PK':<5} {'Nullable':<10} {'Pattern'}")
        print("  " + "─" * 65)
        for col in schema:
            name     = col.get("name", "")
            col_type = col.get("type", "string")
            is_pk    = "PK" if col.get("primary_key") else ""
            nullable = "yes" if col.get("nullable") else "no"
            pattern  = col.get("pattern", "") or ""
            print(f"  {name:<25} {col_type:<12} {is_pk:<5} {nullable:<10} {pattern}")
        print("  " + "─" * 65)
    else:
        print("\n  Schema        : Not defined — AI will decide columns")

    print("\n" + "─" * 60)


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