"""
core/response_parser.py

Parses LLM responses (CSV or JSON) into structured Python data.

CSV parsing (primary):
  - Uses csv.reader for correct handling of quoted fields
  - Validates row-level column count — skips malformed rows
  - Strips any header line or trailing commentary

JSON parsing (used for schema discovery only):
  - Strips markdown fences
  - Validates top-level type is a list
"""

import csv as csv_module
import json
from typing import Tuple, List, Optional
from logger.logger import get_logger

logger = get_logger(__name__)


class ResponseParser:

    # ─────────────────────────────────────────────────────────────
    # CSV parsing  (primary — used for data generation)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def parse_csv(response: str, schema: list) -> list:
        """
        Parse a CSV-format LLM response into a list of row dicts.

        - Uses csv.reader to correctly handle quoted fields
        - Skips any row whose column count doesn't match the schema
        - Skips header-looking rows and blank lines
        """
        column_names   = [col["name"] for col in schema]
        expected_cols  = len(column_names)
        text           = ResponseParser._clean(response)

        rows = []
        skipped_count = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip obvious commentary lines
            if line.startswith("#") or line.startswith("//"):
                continue

            try:
                values = next(csv_module.reader([line], quotechar='"'))
            except StopIteration:
                continue

            if len(values) != expected_cols:
                skipped_count += 1
                logger.debug(f"Skipped row (expected {expected_cols} cols, got {len(values)}): {line[:80]}")
                continue

            # Skip header rows: ALL parsed values must match column names
            if [v.strip().lower() for v in values] == [c.lower() for c in column_names]:
                logger.debug(f"Skipped header row: {line[:80]}")
                continue

            rows.append(dict(zip(column_names, values)))

        if skipped_count:
            logger.debug(f"parse_csv: {len(rows)} rows accepted, {skipped_count} rows skipped (column count mismatch)")
        return rows

    @staticmethod
    def parse_csv_partial(response: str, schema: list) -> Tuple[List[dict], Optional[dict]]:
        """
        Salvage as many complete rows as possible from a truncated CSV response.
        Returns (partial_rows, last_row).
        """
        rows = ResponseParser.parse_csv(response, schema)
        last = rows[-1] if rows else None
        logger.debug(f"parse_csv_partial: recovered {len(rows)} rows from truncated response")
        return rows, last

    # ─────────────────────────────────────────────────────────────
    # JSON parsing  (used for schema discovery)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def parse(response: str) -> list:
        """
        Parse a JSON-format LLM response into a list of dicts.
        Strips markdown fences. Raises ValueError on malformed JSON.
        """
        text = ResponseParser._clean(response)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response was truncated or malformed (JSON error: {e})."
            ) from e

        if not isinstance(result, list):
            raise ValueError(
                f"Expected JSON array but got {type(result).__name__}."
            )
        return result

    @staticmethod
    def parse_partial(response: str) -> Tuple[List[dict], dict]:
        """
        Recover complete rows from a truncated JSON response using
        a bracket-depth walker. Returns (partial_rows, last_row).
        """
        text = ResponseParser._clean(response)
        partial_rows = ResponseParser._extract_complete_objects(text)
        last_row = partial_rows[-1] if partial_rows else {}
        logger.debug(f"parse_partial: recovered {len(partial_rows)} rows from truncated JSON")
        return partial_rows, last_row

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _clean(text: str) -> str:
        """Strip markdown fences from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:text.rfind("```")].strip()
        return text

    @staticmethod
    def _extract_complete_objects(text: str) -> List[dict]:
        """Walk text to extract complete top-level JSON objects."""
        rows = []
        i = 0
        n = len(text)
        while i < n:
            start = text.find("{", i)
            if start == -1:
                break
            depth = 0
            in_string = False
            escape_next = False
            for j in range(start, n):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:j + 1])
                            if isinstance(obj, dict):
                                rows.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                break
        return rows
