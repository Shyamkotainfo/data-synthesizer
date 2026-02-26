"""
core/data_quality.py

Post-generation data quality checker.

Runs automatically after every dataset generation and logs a quality report.
Checks:
  1. Row count  — did we get all expected rows?
  2. Null audit — which columns have nulls? Are any nullable=false columns null?
  3. Uniqueness — are uuid/id-type columns fully unique (primary key check)?
  4. Type audit  — do numeric/boolean columns have the right dtype?
  5. Pattern     — are integer columns within specified range (e.g., 18-65)?
"""

import pandas as pd
from typing import Optional, List, Dict
from logger.logger import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """
    Run quality checks on a generated DataFrame against the schema used.
    Returns a structured quality report dict and logs a summary.
    """

    def __init__(self, df: pd.DataFrame, schema: Optional[List[Dict]], expected_rows: int):
        self.df = df
        self.schema = schema or []
        self.expected_rows = expected_rows
        self._schema_map = {col["name"]: col for col in self.schema}

    def run(self) -> Dict:
        """
        Execute all checks and return a consolidated quality report.
        """
        report = {
            "expected_rows":  self.expected_rows,
            "actual_rows":    len(self.df),
            "columns":        list(self.df.columns),
            "row_count_ok":   len(self.df) >= self.expected_rows,
            "null_report":    self._check_nulls(),
            "uniqueness":     self._check_uniqueness(),
            "type_audit":     self._check_types(),
            "pattern_audit":  self._check_patterns(),
            "overall_pass":   True   # set below
        }

        # Fail if any nullable=false column has nulls
        null_violations = [
            col for col, info in report["null_report"].items()
            if info["null_count"] > 0 and not info["nullable"]
        ]

        # Fail if any uuid/id column is not unique
        pk_violations = [
            col for col, info in report["uniqueness"].items()
            if not info["is_unique"]
        ]

        report["null_violations"]   = null_violations
        report["pk_violations"]     = pk_violations
        report["overall_pass"] = (
            report["row_count_ok"]
            and not null_violations
            and not pk_violations
        )

        self._log_report(report)
        return report

    # ─── Check 1: Row Count ──────────────────────────────────────
    def _check_nulls(self) -> Dict:
        result = {}
        for col in self.df.columns:
            null_count = int(self.df[col].isnull().sum())
            schema_col = self._schema_map.get(col, {})
            nullable   = schema_col.get("nullable", True)
            result[col] = {
                "null_count": null_count,
                "null_pct":   round(null_count / max(len(self.df), 1) * 100, 2),
                "nullable":   nullable,
                "pass":       (null_count == 0) or nullable
            }
        return result

    # ─── Check 2: Primary Key Uniqueness ────────────────────────
    def _check_uniqueness(self) -> Dict:
        result = {}
        # Check columns that are uuid-type OR whose name suggests a PK
        pk_indicators = {"uuid", "id", "key", "pk"}

        for col in self.df.columns:
            schema_col = self._schema_map.get(col, {})
            col_type   = schema_col.get("type", "")
            is_pk_col  = (
                col_type == "uuid"
                or any(ind in col.lower() for ind in pk_indicators)
            )

            if is_pk_col:
                total      = len(self.df)
                unique     = int(self.df[col].nunique())
                duplicates = total - unique
                result[col] = {
                    "total":      total,
                    "unique":     unique,
                    "duplicates": duplicates,
                    "is_unique":  duplicates == 0
                }

        return result

    # ─── Check 3: Type Audit ─────────────────────────────────────
    def _check_types(self) -> Dict:
        result = {}
        expected_dtype_map = {
            "integer": "int",
            "float":   "float",
            "boolean": "bool",
        }

        for col in self.df.columns:
            schema_col    = self._schema_map.get(col, {})
            expected_type = schema_col.get("type", "")

            if expected_type in expected_dtype_map:
                actual_dtype  = str(self.df[col].dtype)
                expected_hint = expected_dtype_map[expected_type]
                matches = expected_hint in actual_dtype

                result[col] = {
                    "expected_type": expected_type,
                    "actual_dtype":  actual_dtype,
                    "pass":          matches
                }

        return result

    # ─── Check 4: Pattern / Range Validation ─────────────────────
    def _check_patterns(self) -> Dict:
        result = {}

        for col in self.df.columns:
            schema_col = self._schema_map.get(col, {})
            pattern    = schema_col.get("pattern", "")
            col_type   = schema_col.get("type", "")

            # Only validate numeric range patterns like "18-65"
            if col_type == "integer" and pattern and "-" in pattern:
                try:
                    parts   = pattern.replace(" ", "").split("-")
                    lo, hi  = int(parts[0]), int(parts[1])
                    series  = pd.to_numeric(self.df[col], errors="coerce").dropna()
                    out_low = int((series < lo).sum())
                    out_hi  = int((series > hi).sum())
                    result[col] = {
                        "pattern":      pattern,
                        "out_of_range": out_low + out_hi,
                        "pass":         (out_low + out_hi) == 0
                    }
                except Exception:
                    pass    # skip if pattern parsing fails

        return result

    # ─── Logging ─────────────────────────────────────────────────
    def _log_report(self, report: Dict):
        divider = "─" * 60
        status  = "✅ PASS" if report["overall_pass"] else "⚠️  ISSUES FOUND"

        logger.info(divider)
        logger.info(f"  DATA QUALITY REPORT  [{status}]")
        logger.info(divider)
        logger.info(f"  Rows expected : {report['expected_rows']:,}")
        logger.info(f"  Rows generated: {report['actual_rows']:,}")
        if not report["row_count_ok"]:
            missing = report["expected_rows"] - report["actual_rows"]
            logger.warning(f"  ⚠ Row count short by {missing} (likely duplicates removed)")

        # Null summary
        logger.info(f"  Null audit:")
        for col, info in report["null_report"].items():
            if info["null_count"] > 0:
                flag = "❌" if not info["nullable"] else "ℹ"
                logger.info(
                    f"    {flag} {col}: {info['null_count']} nulls "
                    f"({info['null_pct']}%) — nullable={info['nullable']}"
                )
            else:
                logger.info(f"    ✅ {col}: no nulls")

        # Uniqueness
        if report["uniqueness"]:
            logger.info(f"  Primary key / uniqueness:")
            for col, info in report["uniqueness"].items():
                flag = "✅" if info["is_unique"] else "❌"
                logger.info(
                    f"    {flag} {col}: {info['unique']}/{info['total']} unique "
                    f"({info['duplicates']} duplicates)"
                )

        # Pattern violations
        if report["pattern_audit"]:
            logger.info(f"  Pattern / range checks:")
            for col, info in report["pattern_audit"].items():
                flag = "✅" if info["pass"] else "⚠"
                logger.info(
                    f"    {flag} {col} [{info['pattern']}]: "
                    f"{info['out_of_range']} out-of-range values"
                )

        # Summary of violations
        if report["null_violations"]:
            logger.warning(
                f"  ❌ Nullable violations: {report['null_violations']}"
            )
        if report["pk_violations"]:
            logger.warning(
                f"  ❌ PK not unique: {report['pk_violations']}"
            )

        logger.info(divider)
