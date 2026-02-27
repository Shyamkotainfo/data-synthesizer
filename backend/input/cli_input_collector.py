"""
CLI Input Collector
Responsible ONLY for collecting raw input from CLI.
Supports:
  - Upload CSV or JSON file (schema + sample rows extracted automatically)
  - Upload JSON schema file only (column definitions, no sample data)
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
    print("  1. Upload file (CSV or JSON) — schema + sample rows used as reference")
    print("  2. Upload JSON schema file only — column definitions, no sample rows")
    print("  3. Define columns manually")
    print("  4. Skip — let AI decide")
    print("─" * 50)

    schema_choice = _prompt_choice("Choose schema mode", choices=["1", "2", "3", "4"], default="4")

    schema_file = None
    schema = None
    sample_file = None

    if schema_choice == "1":
        file_path = _prompt("File path (.csv or .json)", required=True)
        if not os.path.exists(file_path):
            print(f"  ⚠ File not found: {file_path}. Falling back to manual entry.")
            schema_choice = "3"
        elif file_path.lower().endswith(".csv"):
            extracted = _extract_from_csv(file_path)
            if extracted:
                schema = extracted["schema"]
                sample_file = file_path
                print(
                    f"\n  ✅ Schema extracted ({len(schema)} columns, "
                    f"{extracted['sample_count']} sample rows will be used as reference)"
                )
                _print_schema_preview(schema)
            else:
                schema_choice = "3"
        elif file_path.lower().endswith(".json"):
            extracted = _extract_from_json(file_path)
            if extracted:
                schema = extracted.get("schema")
                sample_count = extracted.get("sample_count", 0)
                if schema:
                    if sample_count > 0:
                        sample_file = file_path
                        print(
                            f"\n  ✅ Schema extracted ({len(schema)} columns, "
                            f"{sample_count} sample rows will be used as reference)"
                        )
                    else:
                        print(f"\n  ✅ Schema loaded ({len(schema)} columns, schema-only — no sample rows)")
                    _print_schema_preview(schema)
                else:
                    print("  ⚠ Could not extract schema from JSON. Falling back to manual entry.")
                    schema_choice = "3"
            else:
                schema_choice = "3"
        else:
            print("  ⚠ Unsupported file type. Only .csv and .json are supported. Falling back to manual entry.")
            schema_choice = "3"

    if schema_choice == "2":
        schema_file = _prompt("JSON schema file path (.json)", required=True)
        if not os.path.exists(schema_file):
            print(f"  ⚠ File not found: {schema_file}. Falling back to manual entry.")
            schema_choice = "3"
            schema_file = None

    if schema_choice == "3":
        schema = _collect_schema_columns()

    # ─────────────────────────────────────────
    # PRIMARY KEY CONFIRMATION
    # (only when schema columns are known)
    # ─────────────────────────────────────────
    primary_key = None
    if schema:
        auto_pk = _auto_detect_pk(schema)
        col_names = [c["name"] for c in schema]

        print("\n" + "─" * 50)
        print("  PRIMARY KEY")
        print("─" * 50)
        if auto_pk:
            print(f"  Auto-detected primary key: ✦ {auto_pk}")
        else:
            print("  No primary key auto-detected.")

        pk_input = _prompt(
            f"  Primary key column (Enter to {'confirm' if auto_pk else 'skip'}, or type a column name)",
            default=auto_pk or ""
        )

        if pk_input and pk_input in col_names:
            primary_key = pk_input
            print(f"  ✅ Primary key set to: {primary_key}")
        elif pk_input and pk_input not in col_names:
            print(f"  ⚠ Column '{pk_input}' not found in schema. Primary key not set.")
        elif auto_pk:
            primary_key = auto_pk
            print(f"  ✅ Primary key confirmed: {primary_key}")
        else:
            print("  ↩ No primary key set.")

    return {
        "dataset_name": dataset_name or None,
        "rows": rows or None,
        "format": fmt,
        "description": description or None,
        "schema_file": schema_file or None,
        "schema": schema or None,
        "sample_file": sample_file or None,
        "ai_criteria": ai_criteria or None,
        "target_location": target_location or None,
        "primary_key": primary_key or None,
    }


# ─────────────────────────────────────────
# PRIMARY KEY AUTO-DETECTION
# ─────────────────────────────────────────

def _auto_detect_pk(schema: list) -> str | None:
    """
    Detect the most likely primary key column from schema.
    Priority: first uuid column > column named 'id' > first *_id column.
    """
    if not schema:
        return None
    for col in schema:
        if col.get("type", "").lower() == "uuid":
            return col["name"]
    for col in schema:
        if col["name"].lower() == "id":
            return col["name"]
    for col in schema:
        if col["name"].lower().endswith("_id"):
            return col["name"]
    return None


# ─────────────────────────────────────────
# FILE EXTRACTION HELPERS
# ─────────────────────────────────────────

def _extract_from_csv(file_path: str):
    """
    Read a CSV file, infer a column schema, and count available sample rows.
    The sample_file path is passed forward so InputProcessor loads the actual rows.
    Returns dict with 'schema' and 'sample_count', or None on failure.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  ⚠ pandas is required to read CSV files. Run: pip install pandas")
        return None

    try:
        # Auto-detect separator (e.g., semicolon vs comma)
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        print(f"  ⚠ Could not read CSV: {e}")
        return None

    if df.empty:
        print("  ⚠ CSV file is empty.")
        return None

    dtype_map = {
        "object":         "string",
        "int64":          "integer",
        "int32":          "integer",
        "float64":        "float",
        "float32":        "float",
        "bool":           "boolean",
        "datetime64[ns]": "datetime",
    }

    schema = []
    for col_name, dtype in df.dtypes.items():
        col_type = dtype_map.get(str(dtype), "string")

        name_lower = col_name.lower()
        if "email" in name_lower:
            col_type = "email"
        elif "phone" in name_lower or "mobile" in name_lower:
            col_type = "phone"
        elif "date" in name_lower or "time" in name_lower:
            col_type = "date"
        elif name_lower in ("id", "uuid") or name_lower.endswith("_id"):
            col_type = "uuid"
        elif "url" in name_lower or "link" in name_lower:
            col_type = "url"

        has_nulls = df[col_name].isnull().any()
        schema.append({"name": col_name, "type": col_type, "nullable": bool(has_nulls)})

    return {"schema": schema, "sample_count": min(len(df), 10)}


def _extract_from_json(file_path: str):
    """
    Read a JSON file.
    - List of row-objects → infer schema + pass as sample rows.
    - List with 'name'+'type' per item → treat as schema definition only (no sample rows).
    Returns dict with 'schema' and 'sample_count', or None on failure.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠ Could not read JSON file: {e}")
        return None

    if not (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict)):
        print("  ⚠ JSON must be a non-empty array of objects.")
        return None

    first = data[0]

    # Detect schema-definition format: small objects with 'name' and 'type' keys
    if "name" in first and "type" in first and len(first) <= 6:
        return {"schema": data, "sample_count": 0}

    # Actual row data — infer schema from first row
    schema = []
    for k, v in first.items():
        if isinstance(v, bool):
            col_type = "boolean"
        elif isinstance(v, int):
            col_type = "integer"
        elif isinstance(v, float):
            col_type = "float"
        else:
            col_type = "string"
        schema.append({"name": k, "type": col_type, "nullable": False})

    return {"schema": schema, "sample_count": min(len(data), 10)}


def _print_schema_preview(schema):
    """Print a concise schema table after extraction."""
    print(f"\n  {'Column':<22} {'Type':<12} Nullable")
    print("  " + "─" * 44)
    for col in schema:
        nullable = "yes" if col.get("nullable") else "no"
        print(f"  {col['name']:<22} {col['type']:<12} {nullable}")
    print("  " + "─" * 44)


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

        column = {"name": name, "type": col_type, "nullable": nullable}
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
    suffix = f" [{default}]" if default else ""
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