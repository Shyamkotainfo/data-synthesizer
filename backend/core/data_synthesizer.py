import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from logger.logger import get_logger
from config.settings import get_settings
from llm.llm import LLMQuery
from prompt.dataset_prompt import DatasetPromptBuilder


class DataSynthesizer:

    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        self.llm = LLMQuery()

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

        self.logger.info("Invoking LLM...")

        # -----------------------------
        # STEP 5: Call LLM
        # -----------------------------
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=4000
        )

        data = json.loads(response)
        df = pd.DataFrame(data)

        # -----------------------------
        # STEP 6: Save
        # -----------------------------
        file_path = self._save(
            df=df,
            dataset_name=dataset_name,
            fmt=output_format,
            target_location=target_location
        )

        return {
            "dataset_name": dataset_name,
            "rows_generated": len(df),
            "columns": list(df.columns),
            "file_path": file_path
        }

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

        if output_format not in ["csv", "parquet", "json"]:
            raise ValueError("format must be csv/parquet/json")

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