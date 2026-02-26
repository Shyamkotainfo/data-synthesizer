"""
core/schema_resolver.py

Resolves a schema definition from multiple possible sources:
  1. Explicit schema list (direct dict input)
  2. JSON schema file on disk
  3. Inferred from sample rows (type guessing)
  4. None (let LLM decide)
"""

import json
from typing import List, Dict, Optional
from logger.logger import get_logger

logger = get_logger(__name__)


class SchemaResolver:
    """
    Determines the final schema to use for data generation,
    based on priority order: explicit > file > inferred > None.
    """

    @staticmethod
    def resolve(
        schema: Optional[List[Dict]] = None,
        schema_file: Optional[str] = None,
        sample_rows: Optional[List[Dict]] = None
    ) -> Optional[List[Dict]]:
        """
        Return the resolved schema or None if AI should decide.

        Priority:
          1. schema (explicitly passed columns list)
          2. schema_file (path to a .json schema file)
          3. sample_rows (infer types from first row)
          4. None → LLM infers everything from description
        """

        # 1. Explicit schema takes highest priority
        if schema:
            logger.debug(f"Using explicit schema ({len(schema)} columns)")
            return schema

        # 2. Load from JSON file
        if schema_file:
            try:
                with open(schema_file, "r") as f:
                    loaded = json.load(f)
                logger.info(f"Loaded schema from file: {schema_file} ({len(loaded)} columns)")
                return loaded
            except FileNotFoundError:
                raise FileNotFoundError(f"Schema file not found: {schema_file}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in schema file '{schema_file}': {e}")

        # 3. Infer from sample rows
        if sample_rows and len(sample_rows) > 0:
            inferred = [
                {"name": k, "type": SchemaResolver._infer_type(v)}
                for k, v in sample_rows[0].items()
            ]
            logger.info(f"Inferred schema from sample rows ({len(inferred)} columns)")
            return inferred

        # 4. Let LLM decide
        logger.info("No schema provided — LLM will determine columns from description")
        return None

    @staticmethod
    def _infer_type(value) -> str:
        """Map a Python value to a schema type string."""
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "float"
        return "string"
