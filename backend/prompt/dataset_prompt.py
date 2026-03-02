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
            import io
            import csv
            output = io.StringIO()
            writer = csv.writer(output)
            for r in sample_rows[:5]:
                writer.writerow([str(r.get(col["name"], "")) for col in schema])
            sample_block = (
                "\nReference rows (mimic style, values, and distributions below — "
                "do NOT copy them verbatim):\n"
                + output.getvalue().strip()
            )

        description_block = f"\nDataset context: {description}" if description else ""
        criteria_block = f"\nUser Instructions / AI Criteria: {ai_criteria}" if ai_criteria else ""

        prompt = f"""Generate EXACTLY {rows} unique rows of synthetic data for dataset "{dataset_name}".
This is a PARTIAL BATCH: you are generating rows {batch_start_idx} through {batch_end_idx}.
{description_block}{criteria_block}

Column definitions ({num_cols} columns total):
{col_defs_str}
{sample_block}

**CRITICAL RULES — Read carefully or the system will reject your output:**
1. 🛑 STRICT ROW COUNT: You MUST generate EXACTLY {rows} rows. Do NOT stop early. Count them internally to ensure you output precisely {rows} lines of data. Do NOT output {rows - 1} or {rows + 1} rows.
2. 🛑 STRICT UNIQUENESS: Every single row must be unique. The "{pk_col or 'first'}" column is the PRIMARY KEY. You MUST NOT generate duplicate values for this column within this batch.
3. 🛑 NO EXTRA TEXT: Output ONLY raw comma-separated values. No headers, no markdown fences (```), no bullets, no explanations, no "Here are your rows".
4. 🛑 COLUMN COUNT: Each row must have exactly {num_cols} fields mapped to: {col_names_str}.
5. 🛑 CONTINUITY: Primary keys MUST carry over properly across batches. Since you are generating rows {batch_start_idx}-{batch_end_idx}, ensure sequential IDs and data values continue naturally from {batch_start_idx}.
6. 🛑 QUOTING: Strings should NOT be wrapped in quotes unless the string intrinsically contains commas or quotes. (e.g. `user101` ✅, `"user101"` ❌).

Output (raw CSV only, exactly {rows} rows, no header):"""

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