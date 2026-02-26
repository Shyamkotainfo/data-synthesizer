"""
core/data_synthesizer.py

Orchestrates the full synthetic data generation pipeline.
Delegates each responsibility to a focused sub-module:

  SchemaResolver  → determine which column schema to use
  BatchRunner     → parallel LLM calls, retry, post-process
  FileWriter      → save output in the requested format
  DynamoHistory   → persist job record to DynamoDB
"""

import uuid as uuid_lib
from datetime import datetime
from typing import Dict, Any

from logger.logger import get_logger
from config.settings import get_settings
from core.batch_runner import BatchRunner
from core.schema_resolver import SchemaResolver
from core.file_writer import FileWriter
from core.data_quality import DataQualityChecker
from db.dynamo_history import DynamoHistory


class DataSynthesizer:
    """
    Entry point for data generation.
    Call generate(request) — everything else is handled internally.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        self._runner = BatchRunner()
        self._db = DynamoHistory()

    def generate(self, request: Dict[str, Any]) -> Dict:
        """
        Generate a synthetic dataset from the given request dict.

        Required keys:
            dataset_name (str), rows (int), format (str)

        Optional keys:
            description, schema, schema_file, sample_rows,
            ai_criteria, target_location

        Returns:
            {job_id, dataset_name, rows_generated, columns,
             file_path, generated_at}
        """

        # ── Extract request fields ───────────────────────────────
        dataset_name    = request.get("dataset_name")
        rows            = request.get("rows")
        description     = request.get("description")
        schema          = request.get("schema")
        schema_file     = request.get("schema_file")
        sample_rows     = request.get("sample_rows")
        ai_criteria     = request.get("ai_criteria")
        output_format   = request.get("format", "csv")
        target_location = request.get("target_location")

        # ── Validate ─────────────────────────────────────────────
        self._validate(dataset_name, rows, output_format)

        # ── Resolve schema ────────────────────────────────────────
        resolved_schema = SchemaResolver.resolve(
            schema=schema,
            schema_file=schema_file,
            sample_rows=sample_rows
        )

        # ── Schema discovery (when no schema given) ───────────────
        # Run ONE LLM call to define a consistent schema upfront.
        # Without this, each parallel batch invents its own columns
        # → inconsistent column sets → massive NaN when merged.
        if resolved_schema is None:
            resolved_schema = self._discover_schema(
                dataset_name=dataset_name,
                description=description,
                ai_criteria=ai_criteria
            )

        # ── Generate in parallel batches ─────────────────────────
        self.logger.info(
            f"Starting generation: {dataset_name!r} | "
            f"{rows:,} rows | format={output_format}"
        )
        df = self._runner.run(
            rows=rows,
            dataset_name=dataset_name,
            description=description,
            schema=resolved_schema,
            sample_rows=sample_rows,
            ai_criteria=ai_criteria
        )

        # ── Save to disk ─────────────────────────────────────────
        file_path = FileWriter.save(
            df=df,
            dataset_name=dataset_name,
            fmt=output_format,
            output_dir=self.settings.output_dir,
            target_location=target_location
        )

        # ── Data quality checks ───────────────────────────────────
        quality_report = DataQualityChecker(
            df=df,
            schema=resolved_schema,
            expected_rows=rows
        ).run()

        job_id = str(uuid_lib.uuid4())
        generated_at = datetime.now().isoformat()

        try:
            self._db.save_job({
                "job_id":       job_id,
                "dataset_name": dataset_name,
                "rows":         len(df),
                "format":       output_format,
                "columns":      list(df.columns),
                "file_path":    file_path,
                "generated_at": generated_at,
                "status":       "success"
            })
        except Exception as e:
            self.logger.warning(f"DynamoDB save failed (non-fatal): {e}")

        return {
            "job_id":         job_id,
            "dataset_name":   dataset_name,
            "rows_generated": len(df),
            "columns":        list(df.columns),
            "file_path":      file_path,
            "generated_at":   generated_at,
            "quality_report": quality_report,
        }

    # ── Input validation ─────────────────────────────────────────
    def _validate(self, dataset_name: str, rows: int, output_format: str):
        if not dataset_name:
            raise ValueError("dataset_name is required")
        if not rows:
            raise ValueError("rows is required")
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if output_format not in {"csv", "parquet", "json", "tsv"}:
            raise ValueError("format must be one of: csv, parquet, json, tsv")

    # ── Schema discovery ─────────────────────────────────────────
    def _discover_schema(self, dataset_name: str, description=None, ai_criteria=None):
        """
        Ask the LLM to define a consistent schema before batch generation.
        Called only when the user provides no schema.
        Ensures all parallel batches use identical columns → no NaN columns.
        """
        from llm.llm import LLMQuery
        from prompt.dataset_prompt import DatasetPromptBuilder
        import json

        self.logger.info("No schema provided — discovering schema via LLM...")
        prompt = DatasetPromptBuilder.build_schema_discovery(
            dataset_name=dataset_name,
            description=description,
            ai_criteria=ai_criteria
        )
        llm = LLMQuery()
        response = llm.generate(prompt=prompt, temperature=0.1, max_tokens=1000)

        # Strip fences and parse
        text = response.strip()
        if text.startswith("```"):
            text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:text.rfind("```")].strip()

        try:
            schema = json.loads(text)
            if not isinstance(schema, list):
                raise ValueError("Schema discovery returned non-list")
            col_names = [c.get("name") for c in schema]
            self.logger.info(
                f"Schema discovered: {len(schema)} columns → {col_names}"
            )
            return schema
        except Exception as e:
            self.logger.warning(
                f"Schema discovery failed ({e}) — falling back to 5-column default"
            )
            return [
                {"name": "id",         "type": "uuid",    "nullable": False},
                {"name": "name",       "type": "name",    "nullable": False},
                {"name": "email",      "type": "email",   "nullable": False},
                {"name": "created_at", "type": "date",    "nullable": False},
                {"name": "status",     "type": "string",  "nullable": True},
            ]