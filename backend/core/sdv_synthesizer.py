"""
core/sdv_synthesizer.py

SDV (Synthetic Data Vault) wrapper for single-table data generation.

Workflow:
  1. build_metadata(df, schema)  — create SDV Metadata from our schema dict
  2. train(df, metadata, ...)    — fit a synthesizer on sample/seed data
  3. sample(synthesizer, n)      — generate N synthetic rows
  4. save_model / load_model     — persist synthesizer between runs

Supported synthesizers (SDV Community):
  - gaussian_copula  (default) — fast, transparent, good quality
  - ctgan            — GAN-based, higher fidelity, slower
  - tvae             — Variational autoencoder, higher fidelity, slower
  - copula_gan       — Experimental hybrid
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from logger.logger import get_logger

logger = get_logger(__name__)

# SDV column type map — our schema types → SDV sdtype
_SDV_TYPE_MAP = {
    "uuid":     "id",
    "id":       "id",
    "email":    "email",
    "phone":    "phone_number",
    "address":  "address",
    "name":     "name",
    "ssn":      "ssn",
    "integer":  "numerical",
    "float":    "numerical",
    "boolean":  "boolean",
    "date":     "datetime",
    "string":   "categorical",
}

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


class SDVSynthesizer:
    """
    Trains an SDV synthesizer on sample data and generates synthetic rows.
    """

    # ── Public API ────────────────────────────────────────────────

    @staticmethod
    def build_metadata(df: pd.DataFrame, schema: Optional[List[Dict]] = None):
        """
        Build an SDV SingleTableMetadata object.

        If schema is provided: use explicit column types.
        Otherwise:            auto-detect from the DataFrame.
        """
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()

        if schema:
            columns = {}
            primary_key = None

            for col in schema:
                name     = col.get("name")
                our_type = col.get("type", "string").lower()
                sdtype   = _SDV_TYPE_MAP.get(our_type, "categorical")

                col_meta: Dict[str, Any] = {"sdtype": sdtype}

                if sdtype == "numerical":
                    col_meta["computer_representation"] = (
                        "Float64" if our_type == "float" else "Int64"
                    )
                if sdtype == "datetime":
                    col_meta["datetime_format"] = "%Y-%m-%d"

                columns[name] = col_meta

                if sdtype == "id" and primary_key is None:
                    primary_key = name

            metadata.columns = columns
            if primary_key:
                metadata.primary_key = primary_key

            logger.info(
                f"SDV metadata built from schema: {len(columns)} columns "
                f"| primary_key={primary_key}"
            )
        else:
            metadata.detect_from_dataframe(df)
            logger.info("SDV metadata auto-detected from DataFrame")

        return metadata

    @staticmethod
    def cast_dtypes(df: pd.DataFrame, schema: List[Dict]) -> pd.DataFrame:
        """
        Cast DataFrame columns to their correct Python/pandas dtypes based on schema.
        SDV validates column types during fit() — all-string CSVs must be cast first.
        """
        df = df.copy()
        for col in schema:
            name     = col.get("name")
            our_type = col.get("type", "string").lower()
            if name not in df.columns:
                continue
            try:
                if our_type == "integer":
                    df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
                elif our_type == "float":
                    df[name] = pd.to_numeric(df[name], errors="coerce").astype("Float64")
                elif our_type == "boolean":
                    df[name] = df[name].map(
                        lambda v: str(v).strip().lower() in ("true", "1", "yes")
                    ).astype("boolean")
                elif our_type == "date":
                    df[name] = pd.to_datetime(df[name], errors="coerce")
            except Exception as e:
                logger.warning(f"  dtype cast failed for '{name}' ({our_type}): {e}")
        return df

    @staticmethod
    def train(
        df: pd.DataFrame,
        metadata,
        synthesizer_type: str = "gaussian_copula",
        **kwargs
    ):
        """
        Fit an SDV synthesizer on the provided DataFrame.

        Args:
            df:               Sample/seed DataFrame to train on
            metadata:         SDV SingleTableMetadata object
            synthesizer_type: One of gaussian_copula | ctgan | tvae | copula_gan
            **kwargs:         Extra kwargs passed to the synthesizer constructor

        Returns:
            Fitted synthesizer instance
        """
        synthesizer = SDVSynthesizer._make_synthesizer(
            synthesizer_type, metadata, **kwargs
        )
        logger.info(
            f"Training {synthesizer_type} synthesizer on {len(df):,} rows..."
        )
        synthesizer.fit(df)
        logger.info("Training complete ✅")
        return synthesizer

    @staticmethod
    def sample(synthesizer, n_rows: int) -> pd.DataFrame:
        """
        Sample n_rows of synthetic data from a fitted synthesizer.
        """
        logger.info(f"Sampling {n_rows:,} rows from SDV synthesizer...")
        df = synthesizer.sample(num_rows=n_rows)
        logger.info(f"Sampled {len(df):,} rows ✅")
        return df

    @staticmethod
    def save_model(synthesizer, dataset_name: str, synthesizer_type: str) -> str:
        """
        Persist a trained synthesizer to disk.
        Returns the file path.
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        fname = MODELS_DIR / f"{dataset_name}_{synthesizer_type}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(synthesizer, f)
        logger.info(f"Model saved: {fname}")
        return str(fname)

    @staticmethod
    def load_model(dataset_name: str, synthesizer_type: str = "gaussian_copula"):
        """
        Load a previously saved synthesizer from disk.
        Returns the synthesizer, or None if not found.
        """
        fname = MODELS_DIR / f"{dataset_name}_{synthesizer_type}.pkl"
        if not fname.exists():
            return None
        with open(fname, "rb") as f:
            synthesizer = pickle.load(f)
        logger.info(f"Model loaded: {fname}")
        return synthesizer

    @staticmethod
    def list_models() -> List[Dict]:
        """Return saved model filenames and metadata."""
        if not MODELS_DIR.exists():
            return []
        results = []
        for fpath in MODELS_DIR.glob("*.pkl"):
            parts = fpath.stem.rsplit("_", 1)
            results.append({
                "dataset_name":    parts[0] if len(parts) == 2 else fpath.stem,
                "synthesizer_type": parts[1] if len(parts) == 2 else "unknown",
                "path":            str(fpath),
                "size_mb":         round(fpath.stat().st_size / 1e6, 2),
            })
        return results

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _make_synthesizer(synthesizer_type: str, metadata, **kwargs):
        """Instantiate the correct SDV synthesizer class."""
        stype = synthesizer_type.lower().replace("-", "_")

        if stype == "gaussian_copula":
            from sdv.single_table import GaussianCopulaSynthesizer
            return GaussianCopulaSynthesizer(metadata, **kwargs)

        elif stype == "ctgan":
            from sdv.single_table import CTGANSynthesizer
            return CTGANSynthesizer(metadata, **kwargs)

        elif stype == "tvae":
            from sdv.single_table import TVAESynthesizer
            return TVAESynthesizer(metadata, **kwargs)

        elif stype == "copula_gan":
            from sdv.single_table import CopulaGANSynthesizer
            return CopulaGANSynthesizer(metadata, **kwargs)

        else:
            raise ValueError(
                f"Unknown synthesizer_type '{synthesizer_type}'. "
                f"Choose: gaussian_copula | ctgan | tvae | copula_gan"
            )
