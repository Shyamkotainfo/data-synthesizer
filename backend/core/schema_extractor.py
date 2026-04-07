"""
core/schema_extractor.py
─────────────────────────────────────────────────────────────────
Shared utility for extracting schema + sample rows from uploaded
data files. Supports CSV, JSON, and Parquet.

Used by both:
  - fast_api.py  → POST /schema/upload
  - cli_input_collector.py → file-based schema mode
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json
import io


# Pandas dtype → synthetic column type
DTYPE_MAP = {
    "object":          "string",
    "int64":           "integer",
    "int32":           "integer",
    "float64":         "float",
    "float32":         "float",
    "bool":            "boolean",
    "datetime64[ns]":  "datetime",
}


def _infer_col_type(col_name: str, dtype_str: str) -> str:
    """Map a pandas dtype string + column name to our column type vocabulary."""
    col_type = DTYPE_MAP.get(dtype_str, "string")
    name = col_name.lower()

    if "email" in name:
        return "email"
    if "phone" in name or "mobile" in name:
        return "phone"
    if "date" in name or "time" in name:
        return "date"
    if name in ("id", "uuid") or name.endswith("_id"):
        return "uuid"
    if "url" in name or "link" in name:
        return "url"
    return col_type


def _df_to_schema_and_samples(df, max_samples: int = 5) -> dict:
    """Convert a pandas DataFrame to schema + sample rows."""
    columns = []
    for col_name, dtype in df.dtypes.items():
        columns.append({
            "name":     col_name,
            "type":     _infer_col_type(col_name, str(dtype)),
            "nullable": bool(df[col_name].isnull().any()),
        })

    sample_rows = df.head(max_samples).fillna("").to_dict(orient="records")
    return {
        "columns":     columns,
        "sample_rows": sample_rows,
        "total_rows":  len(df),
    }


# ─────────────────────────────────────────────────────────────────
# Public API — extract from file bytes (for FastAPI uploads)
# ─────────────────────────────────────────────────────────────────

def extract_from_bytes(filename: str, contents: bytes, max_samples: int = 5) -> dict:
    """
    Extract schema + sample rows from raw file bytes.

    Args:
        filename:    original filename (used to detect extension)
        contents:    raw file bytes
        max_samples: number of sample rows to return (default 5)

    Returns:
        {
          "file_type":      "csv" | "json" | "parquet",
          "columns":        [...],
          "sample_rows":    [...],
          "total_rows":     int,
          "columns_count":  int,
        }

    Raises:
        ValueError: on unsupported format or parse errors
    """
    import pandas as pd

    name_lower = filename.lower()

    if name_lower.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
        except Exception as e:
            raise ValueError(f"Could not read CSV: {e}")
        if df.empty:
            raise ValueError("CSV file is empty")
        result = _df_to_schema_and_samples(df, max_samples)
        result["file_type"] = "csv"

    elif name_lower.endswith(".parquet"):
        try:
            df = pd.read_parquet(io.BytesIO(contents), engine="pyarrow")
        except Exception as e:
            raise ValueError(f"Could not read Parquet file: {e}")
        if df.empty:
            raise ValueError("Parquet file is empty")
        result = _df_to_schema_and_samples(df, max_samples)
        result["file_type"] = "parquet"

    elif name_lower.endswith(".json"):
        try:
            data = json.loads(contents.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON: {e}")
        result = _extract_from_json_data(data)
        result["file_type"] = "json"

    else:
        raise ValueError(
            f"Unsupported file type '{filename}'. "
            "Upload a .csv, .json, or .parquet file."
        )

    result["columns_count"] = len(result["columns"])
    return result


# ─────────────────────────────────────────────────────────────────
# Public API — extract from file path (for CLI)
# ─────────────────────────────────────────────────────────────────

def extract_from_path(file_path: str, max_samples: int = 10) -> dict | None:
    """
    Extract schema + sample rows from a local file path.

    Returns the same dict as extract_from_bytes, or None on failure
    (prints a user-friendly error message for CLI usage).
    """
    import pandas as pd

    name_lower = file_path.lower()

    try:
        if name_lower.endswith(".csv"):
            df = pd.read_csv(file_path, sep=None, engine="python")
            if df.empty:
                print("  ⚠ CSV file is empty.")
                return None
            result = _df_to_schema_and_samples(df, max_samples)
            result["file_type"] = "csv"

        elif name_lower.endswith(".parquet"):
            df = pd.read_parquet(file_path, engine="pyarrow")
            if df.empty:
                print("  ⚠ Parquet file is empty.")
                return None
            result = _df_to_schema_and_samples(df, max_samples)
            result["file_type"] = "parquet"

        elif name_lower.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            result = _extract_from_json_data(data)
            result["file_type"] = "json"

        else:
            print("  ⚠ Unsupported file type. Only .csv, .json, and .parquet are supported.")
            return None

        result["columns_count"] = len(result["columns"])
        return result

    except Exception as e:
        print(f"  ⚠ Could not read file: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _extract_from_json_data(data) -> dict:
    """Parse JSON list → schema + sample rows."""
    if not (isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict)):
        raise ValueError("JSON must be a non-empty array of objects.")

    first = data[0]

    # Schema-definition format: small objects with 'name' + 'type' keys
    if "name" in first and "type" in first and len(first) <= 6:
        return {
            "columns":    data,
            "sample_rows": [],
            "total_rows": 0,
        }

    # Actual row data — infer schema from first row
    columns = []
    for k, v in first.items():
        if isinstance(v, bool):
            col_type = "boolean"
        elif isinstance(v, int):
            col_type = "integer"
        elif isinstance(v, float):
            col_type = "float"
        else:
            col_type = "string"
        columns.append({"name": k, "type": col_type, "nullable": False})

    return {
        "columns":    columns,
        "sample_rows": data[:10],
        "total_rows": len(data),
    }
