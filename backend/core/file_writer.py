"""
core/file_writer.py

Handles saving a pandas DataFrame to disk in the requested format.
Supported formats: csv, json, parquet, tsv
"""

import os
import pandas as pd
from datetime import datetime
from logger.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {"csv", "json", "parquet", "tsv"}


class FileWriter:
    """Saves a DataFrame to disk in various formats."""

    @staticmethod
    def save(
        df: pd.DataFrame,
        dataset_name: str,
        fmt: str,
        output_dir: str = "./data/output",
        target_location: str = None
    ) -> str:
        """
        Save the DataFrame to disk.

        Args:
            df:              The DataFrame to save.
            dataset_name:    Used as the base filename.
            fmt:             One of: csv, json, parquet, tsv.
            output_dir:      Default output directory (from settings).
            target_location: Optional override for the output directory.

        Returns:
            Absolute path of the saved file.
        """
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Choose from: {SUPPORTED_FORMATS}")

        base_path = target_location or output_dir
        os.makedirs(base_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(base_path, f"{dataset_name}_{timestamp}.{fmt}")

        if fmt == "csv":
            df.to_csv(path, index=False)

        elif fmt == "tsv":
            df.to_csv(path, index=False, sep="\t")

        elif fmt == "json":
            df.to_json(path, orient="records", indent=2)

        elif fmt == "parquet":
            df.to_parquet(path, index=False)

        logger.info(f"File saved: {path}  ({len(df):,} rows, {fmt.upper()})")
        return path
