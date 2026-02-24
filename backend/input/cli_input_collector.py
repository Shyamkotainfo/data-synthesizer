"""
CLI Input Collector
Responsible ONLY for collecting raw input from CLI.
Supports:
  - Schema file path (JSON)
  - Manual column-by-column schema builder
  - Skip schema (let AI decide)
"""

import json
import os

# Supported column types shown to the user
COLUMN_TYPES = [
    "string", "integer", "float", "boolean",
    "date", "datetime", "email", "phone",
    "uuid", "name", "address", "url"
]


def collect_generation_input():
    """
    Collects raw user input interactively.
    Returns raw request dictionary.
    """

    print("\n" + "─" * 50)
    print("  DATASET SETTINGS")
    print("─" * 50)

    dataset_name = _prompt("Dataset Name", required=True)
    rows = _prompt("Number of Rows", required=True)
    fmt = _prompt_choice("Output Format", choices=["csv", "json", "parquet"], default="csv")
    description = _prompt("Description (optional)")
    ai_criteria = _prompt("AI Criteria (optional)")
    target_location = _prompt("Target File Location (optional)")

    print("\n" + "─" * 50)
    print("  SCHEMA DEFINITION")
    print("─" * 50)
    print("  1. Upload schema file (JSON)")
    print("  2. Define columns manually")
    print("  3. Skip — let AI decide")
    print("─" * 50)

    schema_choice = _prompt_choice("Choose schema mode", choices=["1", "2", "3"], default="3")

    schema_file = None
    schema = None

    if schema_choice == "1":
        schema_file = _prompt("Schema file path (.json)", required=True)
        if not os.path.exists(schema_file):
            print(f"  ⚠ File not found: {schema_file}. Falling back to manual entry.")
            schema_choice = "2"

    if schema_choice == "2":
        schema = _collect_schema_columns()

    return {
        "dataset_name": dataset_name or None,
        "rows": rows or None,
        "format": fmt,
        "description": description or None,
        "schema_file": schema_file or None,
        "schema": schema or None,
        "sample_file": None,
        "ai_criteria": ai_criteria or None,
        "target_location": target_location or None
    }


# ─────────────────────────────────────────
# SCHEMA COLUMN BUILDER
# ─────────────────────────────────────────

def _collect_schema_columns():
    """Interactively collect column definitions one by one."""

    columns = []
    print("\n  Define your columns. Press Enter with no name to finish.\n")

    field_num = 1
    while True:
        print(f"  ── Field {field_num} ──")
        name = _prompt("  Column name (blank to finish)")

        if not name:
            if field_num == 1:
                print("  No columns defined. Skipping schema.")
                return None
            break

        col_type = _prompt_choice(
            "  Type",
            choices=COLUMN_TYPES,
            default="string",
            show_choices=True
        )

        nullable_input = _prompt("  Nullable? (y/n)", default="n").lower()
        nullable = nullable_input in ("y", "yes")

        pattern = _prompt("  Pattern / regex (optional, e.g. [A-Z]{3}-[0-9]{4})")

        column = {
            "name": name,
            "type": col_type,
            "nullable": nullable
        }
        if pattern:
            column["pattern"] = pattern

        columns.append(column)
        print(f"  ✅ Added: {name} ({col_type}{'  nullable' if nullable else ''})\n")
        field_num += 1

    print(f"\n  Schema defined with {len(columns)} column(s):")
    for c in columns:
        flags = []
        if c.get("nullable"):
            flags.append("nullable")
        if c.get("pattern"):
            flags.append(f"pattern={c['pattern']}")
        print(f"    • {c['name']} [{c['type']}]" + (f"  ({', '.join(flags)})" if flags else ""))

    return columns


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def _prompt(label, required=False, default=None):
    """Display a prompt and return stripped input."""
    suffix = ""
    if default:
        suffix = f" [{default}]"
    while True:
        value = input(f"  {label}{suffix}: ").strip()
        if not value and default:
            return default
        if not value and required:
            print(f"  ⚠ '{label}' is required.")
            continue
        return value or None


def _prompt_choice(label, choices, default=None, show_choices=False):
    """Prompt user to pick from a list of choices."""
    if show_choices:
        print(f"  Available types: {', '.join(choices)}")
    suffix = f" [{default}]" if default else f" ({'/'.join(choices)})"
    while True:
        value = input(f"  {label}{suffix}: ").strip().lower()
        if not value and default:
            return default
        if value in choices:
            return value
        print(f"  ⚠ Choose one of: {', '.join(choices)}")