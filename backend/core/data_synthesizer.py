import json
import os
import math
import uuid as uuid_lib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Max rows per single LLM call — keeps response within token limits
BATCH_SIZE = 30

from logger.logger import get_logger
from config.settings import get_settings
from llm.llm import LLMQuery
from prompt.dataset_prompt import DatasetPromptBuilder
from db.dynamo_history import DynamoHistory


class DataSynthesizer:

    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        self.llm = LLMQuery()
        self._db = DynamoHistory()

    # -------------------------------------------------
    # PUBLIC ENTRY
    # -------------------------------------------------
    def generate(self, request: Dict[str, Any]):

        # -----------------------------
        # STEP 1: Extract Fields
        # -----------------------------
        dataset_name = request.get("dataset_name")
        rows = request.get("rows")
        description = request.get("description")
        schema = request.get("schema")
        sample_schema_file = request.get("sample_schema_file")
        sample_rows = request.get("sample_rows")
        ai_criteria = request.get("ai_criteria")
        output_format = request.get("format")
        target_location = request.get("target_location")

        # -----------------------------
        # STEP 2: Validate Mandatory
        # -----------------------------
        self._validate(dataset_name, rows, output_format)

        # -----------------------------
        # STEP 3: Resolve Schema Priority
        # Priority:
        # 1. schema (direct input)
        # 2. sample_schema_file
        # 3. infer from sample_rows
        # -----------------------------
        resolved_schema = self._resolve_schema(
            schema=schema,
            sample_schema_file=sample_schema_file,
            sample_rows=sample_rows
        )

        # -----------------------------
        # STEP 4: Build Prompt
        # -----------------------------
        prompt = DatasetPromptBuilder.build(
            dataset_name=dataset_name,
            rows=rows,
            description=description,
            schema=resolved_schema,
            sample_rows=sample_rows,
            ai_criteria=ai_criteria
        )

        self.logger.info(f"Invoking LLM... ({rows} rows, batch_size={BATCH_SIZE})")

        # -----------------------------
        # STEP 5: Batch LLM Calls
        # -----------------------------
        df = self._generate_in_batches(
            rows=rows,
            dataset_name=dataset_name,
            description=description,
            resolved_schema=resolved_schema,
            sample_rows=sample_rows,
            ai_criteria=ai_criteria
        )

        # -----------------------------
        # STEP 6: Save File
        # -----------------------------
        file_path = self._save(
            df=df,
            dataset_name=dataset_name,
            fmt=output_format,
            target_location=target_location
        )

        # -----------------------------
        # STEP 7: Persist to DynamoDB
        # -----------------------------
        job_id = str(uuid_lib.uuid4())
        generated_at = datetime.now().isoformat()

        job_record = {
            "job_id":       job_id,
            "dataset_name": dataset_name,
            "rows":         len(df),
            "format":       output_format,
            "columns":      list(df.columns),
            "file_path":    file_path,
            "generated_at": generated_at,
            "status":       "success"
        }
        try:
            self._db.save_job(job_record)
        except Exception as e:
            self.logger.warning(f"DynamoDB save failed (non-fatal): {e}")

        return {
            "job_id":         job_id,
            "dataset_name":   dataset_name,
            "rows_generated": len(df),
            "columns":        list(df.columns),
            "file_path":      file_path,
            "generated_at":   generated_at,
        }

    # -------------------------------------------------
    # BATCH GENERATION
    # -------------------------------------------------
    def _generate_in_batches(self, rows, dataset_name, description,
                              resolved_schema, sample_rows, ai_criteria):
        """Split large requests into batches of BATCH_SIZE rows each."""
        num_batches = math.ceil(rows / BATCH_SIZE)
        all_frames = []

        for batch_num in range(1, num_batches + 1):
            batch_rows = min(BATCH_SIZE, rows - (batch_num - 1) * BATCH_SIZE)
            self.logger.info(f"  Batch {batch_num}/{num_batches}: generating {batch_rows} rows...")

            prompt = DatasetPromptBuilder.build(
                dataset_name=dataset_name,
                rows=batch_rows,
                description=description,
                schema=resolved_schema,
                sample_rows=sample_rows,
                ai_criteria=ai_criteria
            )

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=5000
            )

            batch_data = self._parse_llm_response(response)
            all_frames.append(pd.DataFrame(batch_data))

        return pd.concat(all_frames, ignore_index=True)

    # -------------------------------------------------
    # PARSE LLM RESPONSE
    # -------------------------------------------------
    def _parse_llm_response(self, response: str):
        """
        Strip markdown code fences if present, then parse JSON.
        Handles responses like:
            ```json
            [ {...} ]
            ```
        """
        text = response.strip()

        # Remove opening fence (```json or ```)
        if text.startswith("```"):
            # Drop the first line (the fence line)
            text = text[text.find("\n") + 1:]

        # Remove closing fence
        if text.endswith("```"):
            text = text[:text.rfind("```")].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Likely truncated — give a helpful error
            raise ValueError(
                f"LLM response was truncated or malformed (JSON error: {e}). "
                f"Try reducing the number of rows per request or simplify the schema."
            ) from e

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    def _validate(self, dataset_name, rows, output_format):

        if not dataset_name:
            raise ValueError("dataset_name is mandatory")

        if not rows:
            raise ValueError("rows is mandatory")

        if not isinstance(rows, int):
            raise ValueError("rows must be integer")

        if output_format not in ["csv", "parquet", "json", "tsv"]:
            raise ValueError("format must be csv/parquet/json/tsv")

    # -------------------------------------------------
    # SCHEMA RESOLUTION
    # -------------------------------------------------
    def _resolve_schema(self,
                        schema: List[Dict] = None,
                        sample_schema_file: str = None,
                        sample_rows: List[Dict] = None):

        if schema:
            return schema

        if sample_schema_file:
            with open(sample_schema_file, "r") as f:
                return json.load(f)

        if sample_rows:
            # Infer schema from sample rows
            return [
                {"name": k, "type": type(v).__name__}
                for k, v in sample_rows[0].items()
            ]

        return None

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------
    def _save(self, df, dataset_name, fmt, target_location=None):

        base_path = target_location or self.settings.output_dir
        os.makedirs(base_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{base_path}/{dataset_name}_{timestamp}.{fmt}"

        if fmt == "csv":
            df.to_csv(path, index=False)

        elif fmt == "parquet":
            df.to_parquet(path, index=False)

        elif fmt == "json":
            df.to_json(path, orient="records", indent=2)

        return path