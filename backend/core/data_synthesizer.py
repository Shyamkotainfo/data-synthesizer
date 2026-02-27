import csv
import io
import json
import os
import math
import uuid as uuid_lib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Batch size tiers: more columns → smaller batches to stay within token limits
# (1-10 cols → 30, 11-20 → 20, 21-35 → 15, 36-50 → 10, 51-75 → 7, 76-100 → 5)
def _get_batch_size(num_columns: int) -> int:
    """Return rows-per-LLM-call based on schema width."""
    if num_columns <= 10:
        return 30
    elif num_columns <= 20:
        return 20
    elif num_columns <= 35:
        return 15
    elif num_columns <= 50:
        return 10
    elif num_columns <= 75:
        return 7
    else:  # up to 100 columns
        return 5

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
        primary_key = request.get("primary_key")  # confirmed PK column name

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

        num_columns = len(resolved_schema) if resolved_schema else 0
        batch_size = _get_batch_size(num_columns)
        self.logger.info(
            f"Invoking LLM... ({rows} rows, columns={num_columns or 'AI-decided'}, batch_size={batch_size})"
        )

        # -----------------------------
        # STEP 5: Batch LLM Calls
        # -----------------------------
        df = self._generate_in_batches(
            rows=rows,
            dataset_name=dataset_name,
            description=description,
            resolved_schema=resolved_schema,
            sample_rows=sample_rows,
            ai_criteria=ai_criteria,
            batch_size=batch_size,
            primary_key=primary_key
        )

        # -----------------------------
        # STEP 5.5: Quality Report
        # -----------------------------
        quality_report = None
        if primary_key and primary_key in df.columns:
            pk_series = df[primary_key]
            
            # Count exact matches
            duplicate_count = pk_series.duplicated().sum()
            
            # Count nulls or empty strings
            null_count = pk_series.isna().sum() + (pk_series == "").sum()
            
            quality_report = {
                "primary_key_column": primary_key,
                "duplicates_found": int(duplicate_count),
                "nulls_found": int(null_count),
                "is_valid": bool(duplicate_count == 0 and null_count == 0)
            }
            
            if quality_report["is_valid"]:
                self.logger.info(f"✅ Quality Check Passed: 0 duplicates, 0 nulls in PK '{primary_key}'")
            else:
                self.logger.warning(
                    f"⚠ Quality Issues Found: {duplicate_count} duplicates, {null_count} nulls in PK '{primary_key}'"
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

        result = {
            "job_id":         job_id,
            "dataset_name":   dataset_name,
            "rows_generated": len(df),
            "columns":        list(df.columns),
            "file_path":      file_path,
            "generated_at":   generated_at,
        }
        if quality_report:
            result["quality_report"] = quality_report
            
        return result

    # -------------------------------------------------
    # BATCH GENERATION
    # -------------------------------------------------
    def _generate_in_batches(self, rows, dataset_name, description,
                              resolved_schema, sample_rows, ai_criteria,
                              batch_size=20, primary_key=None):
        """Split large requests into column-aware batches to stay within token limits."""
        num_batches = math.ceil(rows / batch_size)
        all_frames = []
        use_csv = resolved_schema is not None   # CSV mode when schema known


        # Keep track of globally seen PKs so we don't allow duplicates *across* batches either
        seen_pks = set()

        for batch_num in range(1, num_batches + 1):
            target_batch_rows = min(batch_size, rows - (batch_num - 1) * batch_size)
            batch_start_idx = ((batch_num - 1) * batch_size) + 1
            batch_end_idx = batch_start_idx + target_batch_rows - 1

            self.logger.info(f"  Batch {batch_num}/{num_batches}: generating rows {batch_start_idx}-{batch_end_idx}...")

            rows_needed = target_batch_rows
            batch_frames = []
            retry_count = 0
            max_retries = 3

            while rows_needed > 0 and retry_count <= max_retries:
                if retry_count > 0:
                    self.logger.info(f"    Retry {retry_count}/{max_retries}: regenerating {rows_needed} missing/duplicate rows...")

                prompt = DatasetPromptBuilder.build(
                    dataset_name=dataset_name,
                    rows=rows_needed,
                    description=description,
                    schema=resolved_schema,
                    sample_rows=sample_rows,
                    ai_criteria=ai_criteria,
                    primary_key=primary_key,
                    batch_start_idx=batch_start_idx + (target_batch_rows - rows_needed),
                    batch_end_idx=batch_end_idx
                )

                response = self.llm.generate(
                    prompt=prompt,
                    temperature=0.3 + (0.1 * retry_count), # slightly increase temp on retry
                    max_tokens=8000
                )

                try:
                    batch_data = self._parse_llm_response(
                        response,
                        schema=resolved_schema if use_csv else None
                    )
                    df_chunk = pd.DataFrame(batch_data)
                    
                    if primary_key and primary_key in df_chunk.columns:
                        # Drop nulls
                        df_chunk = df_chunk.dropna(subset=[primary_key])
                        df_chunk = df_chunk[df_chunk[primary_key] != ""]
                        
                        # Drop intra-chunk duplicates
                        df_chunk = df_chunk.drop_duplicates(subset=[primary_key], keep='first')
                        
                        # Drop cross-batch duplicates
                        df_chunk = df_chunk[~df_chunk[primary_key].isin(seen_pks)]
                        
                        # Update seen set
                        seen_pks.update(df_chunk[primary_key].tolist())

                    batch_frames.append(df_chunk)
                    rows_needed -= len(df_chunk)

                except Exception as e:
                    self.logger.warning(f"    Parse error during generation: {e}")
                
                retry_count += 1

            if rows_needed > 0:
                self.logger.warning(f"    Could not safely generate {rows_needed} rows after {max_retries} retries. Proceeding with fewer rows.")

            if batch_frames:
                all_frames.append(pd.concat(batch_frames, ignore_index=True))

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    # -------------------------------------------------
    # PARSE LLM RESPONSE
    # -------------------------------------------------
    def _parse_llm_response(self, response: str, schema: list = None):
        """
        Parse the LLM response.
        - schema provided → expect raw CSV rows, map to column names from schema.
        - schema is None  → expect a JSON array of objects (AI-decided columns).
        Strips markdown code fences in both cases.
        """
        text = response.strip()

        # Strip opening code fence (```csv, ```json, ```)
        if text.startswith("```"):
            text = text[text.find("\n") + 1:]

        # Strip closing code fence
        if text.endswith("```"):
            text = text[:text.rfind("```")].strip()

        # ── CSV mode ────────────────────────────────────
        if schema is not None:
            columns = [col["name"] for col in schema]
            num_cols = len(columns)
            reader = csv.reader(io.StringIO(text))
            rows = []
            for line in reader:
                if not line:          # skip blank lines
                    continue
                # Discard any header-like row the LLM might have emitted
                if line[0].strip().lower() == columns[0].lower():
                    continue
                # Only accept rows with the exact expected column count
                if len(line) != num_cols:
                    self.logger.warning(
                        f"Skipping malformed row (got {len(line)} fields, expected {num_cols}): {line}"
                    )
                    continue
                rows.append(dict(zip(columns, [v.strip() for v in line])))

            if not rows:
                raise ValueError(
                    "LLM returned no valid CSV rows. "
                    "Check that the schema column count matches the model output."
                )
            return rows

        # ── JSON mode (no schema / AI-decided columns) ──
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM response was truncated or malformed (JSON error: {e}). "
                f"Try reducing rows per request or simplifying the schema."
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