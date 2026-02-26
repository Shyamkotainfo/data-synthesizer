"""
core/batch_runner.py

Handles parallel LLM batch execution for large dataset generation.

Responsibilities:
  - Split total row count into BATCH_SIZE chunks
  - Run batches concurrently via ThreadPoolExecutor
  - Retry failed batches with exponential backoff
  - Post-process merged DataFrame (UUID uniqueness, dedup)
  - Report real-time progress
"""

import math
import time
import uuid as uuid_lib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from llm.llm import LLMQuery
from prompt.dataset_prompt import DatasetPromptBuilder
from core.response_parser import ResponseParser
from logger.logger import get_logger

logger = get_logger(__name__)

# ─── Tuning constants ──────────────────────────────────────────
BATCH_SIZE   = 30    # default rows per LLM call (safe for schemas up to ~6 cols)
MAX_WORKERS  = 20    # concurrent Bedrock API calls
MAX_RETRIES  = 3     # retries per batch on failure
RETRY_BACKOFF = 2    # base seconds; doubles each attempt


def _dynamic_batch_size(schema) -> int:
    """
    CSV is ~3x more compact than JSON, so we can afford much larger batches.
    Still reduce for very wide schemas to stay within Nova Pro's 5120 token limit.

      cols >  8  → 40 rows/batch
      cols >  5  → 60 rows/batch
      cols <= 5  → 80 rows/batch
      no schema   → 50 rows/batch (conservative)
    """
    if not schema:
        return 50
    n = len(schema)
    if n > 8:
        return 40
    if n > 5:
        return 60
    return 80


class BatchRunner:
    """
    Executes LLM data generation in parallel batches and
    merges the results into a single DataFrame.
    """

    def __init__(self):
        self.llm = LLMQuery()

    def run(
        self,
        rows: int,
        dataset_name: str,
        description: Optional[str],
        schema: Optional[List[Dict]],
        sample_rows: Optional[List[Dict]],
        ai_criteria: Optional[str]
    ) -> pd.DataFrame:
        """
        Generate `rows` rows of data using parallel LLM batches.

        Returns:
            A merged, post-processed DataFrame with exactly `rows` rows.
        """
        batch_size = _dynamic_batch_size(schema)
        num_batches = math.ceil(rows / batch_size)
        logger.info(
            f"BatchRunner: {rows:,} rows | "
            f"{num_batches} batches | {MAX_WORKERS} workers | "
            f"{batch_size} rows/batch"
        )
        start_time = time.time()

        # Build list of (batch_num, batch_row_count)
        batch_plan = [
            (i, min(batch_size, rows - (i - 1) * batch_size))
            for i in range(1, num_batches + 1)
        ]

        all_frames: List[Optional[pd.DataFrame]] = [None] * num_batches
        completed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    self._run_single_batch,
                    batch_num, batch_rows,
                    dataset_name, description, schema,
                    sample_rows, ai_criteria, num_batches
                ): (idx, batch_num)
                for idx, (batch_num, batch_rows) in enumerate(batch_plan)
            }

            for future in as_completed(future_map):
                idx, batch_num = future_map[future]
                try:
                    all_frames[idx] = future.result()
                    completed += 1
                    elapsed = time.time() - start_time
                    rows_done = completed * BATCH_SIZE
                    rate = int(rows_done / elapsed) if elapsed > 0 else 0
                    pct = int(completed / num_batches * 100)
                    logger.info(
                        f"  [{pct:3d}%] batch {batch_num}/{num_batches} done "
                        f"| ~{rate:,} rows/sec | {elapsed:.0f}s elapsed"
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed permanently: {e}")
                    raise

        merged = pd.concat(
            [f for f in all_frames if f is not None],
            ignore_index=True
        )
        total_time = time.time() - start_time
        logger.info(
            f"BatchRunner complete: {len(merged):,} rows in {total_time:.1f}s "
            f"(~{int(len(merged) / total_time):,} rows/sec)"
        )

        return self._post_process(merged, schema)

    # ─────────────────────────────────────────────────────────────
    # Single batch execution with retry
    # ─────────────────────────────────────────────────────────────
    def _run_single_batch(
        self,
        batch_num: int,
        batch_rows: int,
        dataset_name: str,
        description: Optional[str],
        schema: Optional[List[Dict]],
        sample_rows: Optional[List[Dict]],
        ai_criteria: Optional[str],
        total_batches: int
    ) -> pd.DataFrame:
        """
        Run one LLM batch.
        On truncated response: salvage partial rows and ask LLM to continue
        from the last complete row instead of retrying the whole batch.
        """
        prompt = DatasetPromptBuilder.build(
            dataset_name=dataset_name,
            rows=batch_rows,
            description=description,
            schema=schema,
            sample_rows=sample_rows,
            ai_criteria=ai_criteria,
            batch_num=batch_num,
            total_batches=total_batches
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=5000
                )
                rows_data = ResponseParser.parse_csv(response, schema)
                if not rows_data:
                    raise ValueError("No valid rows parsed from LLM response")
                return pd.DataFrame(rows_data)

            except ValueError as e:
                # Truncated / malformed response — salvage partial CSV rows
                partial_rows, last_row = ResponseParser.parse_csv_partial(response, schema) if schema else ([], {})

                if partial_rows and len(partial_rows) < batch_rows:
                    still_needed = batch_rows - len(partial_rows)
                    logger.warning(
                        f"Batch {batch_num}: truncated after {len(partial_rows)} rows. "
                        f"Continuing for remaining {still_needed} rows..."
                    )
                    cont_prompt = DatasetPromptBuilder.build_continuation(
                        dataset_name=dataset_name,
                        rows=still_needed,
                        description=description,
                        schema=schema,
                        ai_criteria=ai_criteria,
                        last_row=last_row,
                        batch_num=batch_num,
                        total_batches=total_batches
                    )
                    try:
                        cont_response = self.llm.generate(
                            prompt=cont_prompt,
                            temperature=0.3,
                            max_tokens=5000
                        )
                        cont_rows = ResponseParser.parse_csv(cont_response, schema)
                        all_rows = partial_rows + cont_rows
                        logger.info(
                            f"Batch {batch_num}: continuation — "
                            f"{len(partial_rows)} + {len(cont_rows)} = {len(all_rows)} rows"
                        )
                        return pd.DataFrame(all_rows)
                    except Exception:
                        if partial_rows:
                            logger.warning(
                                f"Batch {batch_num}: keeping {len(partial_rows)} partial rows"
                            )
                            return pd.DataFrame(partial_rows)

                # No partial rows salvaged — retry whole batch
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        f"Batch {batch_num} attempt {attempt}: no partial rows, "
                        f"retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    # Reset to original prompt for full retry
                    prompt = DatasetPromptBuilder.build(
                        dataset_name=dataset_name,
                        rows=batch_rows,
                        description=description,
                        schema=schema,
                        sample_rows=sample_rows,
                        ai_criteria=ai_criteria,
                        batch_num=batch_num,
                        total_batches=total_batches
                    )
                else:
                    logger.error(f"Batch {batch_num} failed after {MAX_RETRIES} attempts.")
                    raise

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(f"Batch {batch_num} attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Batch {batch_num} failed after {MAX_RETRIES} attempts.")
                    raise

    # ─────────────────────────────────────────────────────────────
    # Post-processing: uniqueness + dedup
    # ─────────────────────────────────────────────────────────────
    def _post_process(
        self,
        df: pd.DataFrame,
        schema: Optional[List[Dict]]
    ) -> pd.DataFrame:
        """
        After merging all batches:
        1. Replace uuid-type columns with Python-generated UUIDs (100% unique)
        2. Drop exact duplicate rows
        """
        if schema:
            uuid_cols = [
                col["name"] for col in schema
                if col.get("type") == "uuid" and col["name"] in df.columns
            ]
            for col in uuid_cols:
                df[col] = [str(uuid_lib.uuid4()) for _ in range(len(df))]
                logger.info(
                    f"  Replaced '{col}' with {len(df):,} guaranteed-unique UUIDs"
                )

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  Removed {removed:,} duplicate row(s)")

        return df
