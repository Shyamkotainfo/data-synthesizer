"""
core/data_synthesizer.py

Orchestrates the full synthetic data generation pipeline.

Supports two modes:
  mode='sdv' (default) — Train an SDV statistical model on seed data,
                         then sample N rows instantly.
  mode='llm'           — Parallel LLM batch generation (legacy pipeline).

SDV flow:
  1. Resolve / discover schema
  2. Get seed data (user-provided sample OR LLM-generated 50-row seed)
  3. Build SDV Metadata from schema
  4. Train synthesizer (gaussian_copula | ctgan | tvae | copula_gan)
  5. Sample N rows
  6. Post-process (UUID replacement, dedup)
  7. Quality check → save file → DynamoDB

LLM flow (unchanged):
  1-3: same as above
  4. BatchRunner parallel LLM calls
  5-7: same
"""

import uuid as uuid_lib
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from logger.logger import get_logger
from config.settings import get_settings
from core.schema_resolver import SchemaResolver
from core.file_writer import FileWriter
from core.data_quality import DataQualityChecker
from db.dynamo_history import DynamoHistory


class DataSynthesizer:
    """Entry point for data generation. Call generate(request)."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        self._db = DynamoHistory()

    # ── Public API ────────────────────────────────────────────────
    def generate(self, request: Dict[str, Any]) -> Dict:
        """
        Generate a synthetic dataset.

        Key request fields:
            dataset_name (str)   — required
            rows         (int)   — required
            format       (str)   — csv | json | parquet | tsv (default csv)
            mode         (str)   — 'sdv' (default) | 'llm'
            synthesizer  (str)   — gaussian_copula | ctgan | tvae | copula_gan
            schema       (list)  — optional explicit column definitions
            sample_df    (DataFrame) — optional real data to train SDV on
            description  (str)   — optional dataset description
            ai_criteria  (str)   — optional extra LLM criteria
            target_location (str) — optional output path override
        """
        dataset_name    = request.get("dataset_name")
        rows            = request.get("rows")
        description     = request.get("description")
        schema          = request.get("schema")
        schema_file     = request.get("schema_file")
        sample_rows     = request.get("sample_rows")        # for LLM mode
        sample_df       = request.get("sample_df")          # for SDV mode (DataFrame)
        ai_criteria     = request.get("ai_criteria")
        output_format   = request.get("format", "csv")
        target_location = request.get("target_location")
        mode            = request.get("mode", "sdv").lower()
        synthesizer_type = request.get("synthesizer", "gaussian_copula")

        self._validate(dataset_name, rows, output_format, mode)

        # ── Resolve schema ────────────────────────────────────────
        resolved_schema = SchemaResolver.resolve(
            schema=schema,
            schema_file=schema_file,
            sample_rows=sample_rows
        )
        if resolved_schema is None:
            resolved_schema = self._discover_schema(
                dataset_name=dataset_name,
                description=description,
                ai_criteria=ai_criteria
            )

        # ── Route by mode ─────────────────────────────────────────
        self.logger.info(
            f"[{mode.upper()}] Generating {dataset_name!r} | "
            f"{rows:,} rows | format={output_format}"
        )

        if mode == "sdv":
            df = self._generate_sdv(
                rows=rows,
                dataset_name=dataset_name,
                description=description,
                schema=resolved_schema,
                sample_df=sample_df,
                ai_criteria=ai_criteria,
                synthesizer_type=synthesizer_type
            )
        else:
            df = self._generate_llm(
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

        # ── Quality checks ────────────────────────────────────────
        quality_report = DataQualityChecker(
            df=df,
            schema=resolved_schema,
            expected_rows=rows
        ).run()

        # ── Persist to DynamoDB ───────────────────────────────────
        job_id = str(uuid_lib.uuid4())
        generated_at = datetime.now().isoformat()
        try:
            self._db.save_job({
                "job_id":           job_id,
                "dataset_name":     dataset_name,
                "rows":             len(df),
                "format":           output_format,
                "columns":          list(df.columns),
                "file_path":        file_path,
                "generated_at":     generated_at,
                "mode":             mode,
                "synthesizer_type": synthesizer_type if mode == "sdv" else "llm",
                "status":           "success"
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
            "mode":           mode,
            "quality_report": quality_report,
        }

    # ── SDV generation ────────────────────────────────────────────
    def _generate_sdv(
        self,
        rows: int,
        dataset_name: str,
        description: Optional[str],
        schema,
        sample_df: Optional[pd.DataFrame],
        ai_criteria: Optional[str],
        synthesizer_type: str
    ) -> pd.DataFrame:
        """
        SDV pipeline:
          1. Get seed data (user-provided OR LLM-generated 50-row seed)
          2. Build SDV Metadata
          3. Train synthesizer
          4. Sample N rows
          5. Post-process (UUID, dedup)
        """
        from core.sdv_synthesizer import SDVSynthesizer

        # Step 1 — seed data
        if sample_df is not None and len(sample_df) >= 50:
            seed_df = sample_df
            self.logger.info(
                f"Using provided sample data: {len(seed_df):,} rows for SDV training"
            )
        else:
            # Tiered seed size — more seed rows = better SDV fidelity
            # scaled to the target without over-spending on LLM calls
            if rows <= 1_000:
                seed_rows = 100
            elif rows <= 10_000:
                seed_rows = 200
            elif rows <= 100_000:
                seed_rows = 500
            elif rows <= 1_000_000:
                seed_rows = 1_000
            else:
                seed_rows = 1_500

            # If sample data exists but is too small (<50 rows), use it to guide the LLM
            llm_sample_rows = None
            if sample_df is not None and len(sample_df) < 50:
                self.logger.info(
                    f"Provided sample data is too small ({len(sample_df)} rows) for SDV training. "
                    f"Using it as few-shot examples for LLM seed generation."
                )
                llm_sample_rows = sample_df.head(5).fillna("").to_dict(orient="records")

            self.logger.info(
                f"SDV seed: {seed_rows} LLM rows for {rows:,} target "
                f"(ratio 1:{rows // seed_rows:,})"
            )
            seed_df = self._generate_llm(
                rows=seed_rows,
                dataset_name=dataset_name,
                description=description,
                schema=schema,
                sample_rows=llm_sample_rows,
                ai_criteria=ai_criteria
            )
            self.logger.info(f"LLM seed generated: {len(seed_df):,} rows")

        # Step 2 — Cast dtypes: CSV seed arrives as all-strings; SDV needs proper types
        if schema:
            seed_df = SDVSynthesizer.cast_dtypes(seed_df, schema)
            self.logger.info("Seed data dtypes cast to match schema")

        # Step 2b — Deduplicate seed on primary key (SDV requires unique PK values)
        pk_col = next((col["name"] for col in (schema or []) if col.get("primary_key")), None)
        if pk_col and pk_col in seed_df.columns:
            # Strip whitespace first — LLM sometimes generates values like ' 12345' that
            # Python treats as unique strings but SDV normalises and flags as duplicates
            if seed_df[pk_col].dtype == object:
                seed_df[pk_col] = seed_df[pk_col].str.strip()
            before = len(seed_df)
            seed_df = seed_df.drop_duplicates(subset=[pk_col]).reset_index(drop=True)
            removed = before - len(seed_df)
            self.logger.info(
                f"Seed PK dedup ('{pk_col}'): {before} → {len(seed_df)} rows"
                + (f" ({removed} dupes removed)" if removed else "")
            )

        # Step 3 — SDV Metadata
        metadata = SDVSynthesizer.build_metadata(df=seed_df, schema=schema)

        # Step 4 — Train
        synthesizer = SDVSynthesizer.train(
            df=seed_df,
            metadata=metadata,
            synthesizer_type=synthesizer_type
        )

        # Save model for reuse
        SDVSynthesizer.save_model(synthesizer, dataset_name, synthesizer_type)

        # Step 4 — Sample
        df = SDVSynthesizer.sample(synthesizer, n_rows=rows)

        # Step 5 — Post-process
        return self._post_process(df, schema)

    # ── LLM generation (legacy) ───────────────────────────────────
    def _generate_llm(
        self,
        rows: int,
        dataset_name: str,
        description: Optional[str],
        schema,
        sample_rows,
        ai_criteria: Optional[str]
    ) -> pd.DataFrame:
        """Delegate to BatchRunner (parallel LLM calls)."""
        from core.batch_runner import BatchRunner
        return BatchRunner().run(
            rows=rows,
            dataset_name=dataset_name,
            description=description,
            schema=schema,
            sample_rows=sample_rows,
            ai_criteria=ai_criteria
        )

    # ── Post-processing ───────────────────────────────────────────
    def _post_process(self, df: pd.DataFrame, schema) -> pd.DataFrame:
        """
        Post-process SDV output:
        1. Replace uuid-type columns with guaranteed-unique Python UUIDs.
        2. Drop identical duplicate rows (whole-row).

        NOTE: We do NOT enforce PK uniqueness by dropping rows here.
        SDV is a statistical model — for categorical columns like DMC or ID,
        it can only sample from values it saw during training.  Forcing PK
        uniqueness by dropping rows would delete the vast majority of output.
        Non-unique PKs are reported by the quality checker as a warning.
        """
        if schema:
            # Replace uuid-type columns with guaranteed-unique UUIDs
            uuid_cols = [
                col["name"] for col in schema
                if col.get("type") == "uuid" and col["name"] in df.columns
            ]
            for col in uuid_cols:
                df[col] = [str(uuid_lib.uuid4()) for _ in range(len(df))]
                self.logger.info(
                    f"  Replaced '{col}' with {len(df):,} guaranteed-unique UUIDs"
                )

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            self.logger.info(f"  Removed {removed:,} exact duplicate row(s)")
        return df


    # ── Schema discovery ─────────────────────────────────────────
    def _discover_schema(self, dataset_name: str, description=None, ai_criteria=None):
        """Ask the LLM once to define a consistent schema upfront."""
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
                {"name": "id",         "type": "uuid",   "nullable": False},
                {"name": "name",       "type": "name",   "nullable": False},
                {"name": "email",      "type": "email",  "nullable": False},
                {"name": "created_at", "type": "date",   "nullable": False},
                {"name": "status",     "type": "string", "nullable": True},
            ]

    # ── Validation ────────────────────────────────────────────────
    def _validate(self, dataset_name, rows, output_format, mode):
        if not dataset_name:
            raise ValueError("dataset_name is required")
        if not rows or not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive integer")
        if output_format not in {"csv", "parquet", "json", "tsv"}:
            raise ValueError("format must be one of: csv, parquet, json, tsv")
        if mode not in {"sdv", "llm"}:
            raise ValueError("mode must be 'sdv' or 'llm'")