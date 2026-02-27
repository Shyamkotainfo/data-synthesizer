import json


def _detect_primary_key(schema: list) -> str | None:
    """
    Return the name of the primary key column, or None if not detected.
    Detection rules (in priority order):
      1. First column with type 'uuid'
      2. Column named exactly 'id'
      3. First column whose name ends with '_id'
    """
    if not schema:
        return None

    # Priority 1: first uuid column
    for col in schema:
        if col.get("type", "").lower() == "uuid":
            return col["name"]

    # Priority 2: column named 'id'
    for col in schema:
        if col["name"].lower() == "id":
            return col["name"]

    # Priority 3: first *_id column
    for col in schema:
        if col["name"].lower().endswith("_id"):
            return col["name"]

    return None


class DatasetPromptBuilder:

    @staticmethod
    def build(
        dataset_name,
        rows,
        description=None,
        schema=None,
        sample_rows=None,
        ai_criteria=None,
        primary_key=None,       # explicit PK column name (confirmed by user)
        batch_start_idx=1,      # pagination: which row # we start at
        batch_end_idx=None,     # pagination: which row # we end at
    ):
        """
        Build the LLM prompt.

        - If schema is provided  → ask for CSV output (no header, strict column count).
        - If schema is None      → ask for JSON array (AI decides columns).
        """
        if schema:
            return DatasetPromptBuilder._build_csv_prompt(
                dataset_name=dataset_name,
                rows=rows,
                schema=schema,
                description=description,
                sample_rows=sample_rows,
                ai_criteria=ai_criteria,
                primary_key=primary_key,
                batch_start_idx=batch_start_idx,
                batch_end_idx=batch_end_idx or rows,
            )
        else:
            return DatasetPromptBuilder._build_json_prompt(
                dataset_name=dataset_name,
                rows=rows,
                description=description,
                sample_rows=sample_rows,
                ai_criteria=ai_criteria,
            )

    # ─────────────────────────────────────────────────
    # CSV PROMPT (used when schema is known)
    # ─────────────────────────────────────────────────
    @staticmethod
    def _build_csv_prompt(dataset_name, rows, schema,
                          description=None, sample_rows=None,
                          ai_criteria=None, primary_key=None,
                          batch_start_idx=1, batch_end_idx=None):

        # Use confirmed PK if given, otherwise auto-detect from schema
        pk_col = primary_key or _detect_primary_key(schema)
        num_cols = len(schema)

        # Build column definition lines
        col_defs = []
        for col in schema:
            name = col.get("name", "")
            col_type = col.get("type", "string")
            nullable = col.get("nullable", False)
            pattern = col.get("pattern", "")
            distribution = col.get("distribution", "")

            notes = []
            notes.append("NULL" if nullable else "NOT NULL")
            if name == pk_col:
                notes.append("PRIMARY KEY")
            if pattern:
                notes.append(f"range/pattern: {pattern}")
            if distribution:
                notes.append(f"distribution: {distribution}")

            col_defs.append(f"  {name} ({col_type}, {', '.join(notes)})")

        col_defs_str = "\n".join(col_defs)
        col_names_str = ", ".join(col["name"] for col in schema)

        # Sample rows block
        sample_block = ""
        if sample_rows:
            sample_lines = []
            for r in sample_rows[:5]:
                sample_lines.append(", ".join(str(r.get(col["name"], "")) for col in schema))
            sample_block = (
                "\nReference rows (mimic style, values, and distributions below — "
                "do NOT copy them verbatim):\n"
                + "\n".join(sample_lines)
            )

        description_block = f"\nDataset context: {description}" if description else ""
        criteria_block = f"\nUser Instructions / AI Criteria: {ai_criteria}" if ai_criteria else ""

        prompt = f"""Generate exactly {rows} unique rows of synthetic data for dataset "{dataset_name}".
This is a PARTIAL BATCH: you are generating rows {batch_start_idx} through {batch_end_idx}.
{description_block}{criteria_block}

Column definitions ({num_cols} columns total):
{col_defs_str}
{sample_block}

**Rules — read carefully or generation will fail:**
- 🚫 DO NOT include headers, explanations, summaries, or additional text.
- ✅ Output ONLY comma-separated values. No markdown. No code fences (```). No bullets.
- ✅ Each row must have exactly {num_cols} fields — strictly match the column count in this order: {col_names_str}.
- ✅ Strings should **NOT** be wrapped in quotes unless the string contains commas or quotes that require escaping (e.g. `user101` ✅, not `"user101"` ❌).
- ✅ Primary keys MUST carry over properly across batches. Since you are generating rows {batch_start_idx}-{batch_end_idx}, ensure sequential IDs and data values continue naturally from {batch_start_idx}.
- ✅ Every single row must be unique worldwide. The "{pk_col or 'first'}" column is the PRIMARY KEY.
- ✅ Do NOT add trailing text like "And so on..." or "Here are your rows...".

Output (raw CSV only, no header):"""

        return prompt.strip()

    # ─────────────────────────────────────────────────
    # JSON PROMPT (used when schema is unknown / AI decides)
    # ─────────────────────────────────────────────────
    @staticmethod
    def _build_json_prompt(dataset_name, rows, description=None,
                           sample_rows=None, ai_criteria=None):

        sections = []
        sections.append(f"Generate {rows} rows of realistic tabular data.")
        sections.append(f"Dataset Name: {dataset_name}")

        if description:
            sections.append(f"\nDescription:\n{description}")

        if sample_rows:
            sections.append(f"\nSample Rows:\n{json.dumps(sample_rows, indent=2)}")

        if ai_criteria:
            sections.append(f"\nAI Criteria:\n{ai_criteria}")

        sections.append("""
Rules:
- Return ONLY valid JSON
- Return a JSON array of objects
- No explanation, no markdown fences
- Generate realistic and varied values
- Every row must be unique
""")

        return "\n".join(sections)