import json


class DatasetPromptBuilder:

    @staticmethod
    def build_schema_discovery(dataset_name, description=None, ai_criteria=None):
        """
        Ask the LLM to define a consistent schema for the dataset.
        Called once before batch generation when no schema is provided.
        Returns a prompt whose response should be a JSON array of column dicts.
        """
        parts = [
            f"You are designing a dataset called '{dataset_name}'.",
        ]
        if description:
            parts.append(f"Description: {description}")
        if ai_criteria:
            parts.append(f"Criteria: {ai_criteria}")

        parts.append("""
Define a schema for this dataset. Return ONLY a JSON array of column definitions.
Each column must have:
  - "name": snake_case column name (string)
  - "type": one of uuid, string, integer, float, boolean, date, email, name, phone, address
  - "nullable": true or false

Rules:
- Include a primary key column of type "uuid" named appropriately (e.g. "user_id", "order_id")
- Choose 6-10 meaningful columns relevant to the dataset
- Do NOT include redundant columns (e.g. both "is_active" and "active")
- Return ONLY the JSON array, no explanation, no markdown fences

Example output:
[
  {"name": "user_id",    "type": "uuid",    "nullable": false},
  {"name": "full_name",  "type": "name",    "nullable": false},
  {"name": "email",      "type": "email",   "nullable": false},
  {"name": "age",        "type": "integer", "nullable": false},
  {"name": "city",       "type": "string",  "nullable": true}
]
""")
        return "\n".join(parts)


    @staticmethod
    def build(
        dataset_name,
        rows,
        description=None,
        schema=None,
        sample_rows=None,
        ai_criteria=None,
        batch_num=None,
        total_batches=None
    ):
        # Build column definitions string from schema
        col_defs = DatasetPromptBuilder._render_csv_columns(schema) if schema else ""
        expected_cols = len(schema) if schema else "?"
        col_names_str = ", ".join(c["name"] for c in schema) if schema else ""

        batch_ctx = ""
        if batch_num is not None and total_batches is not None:
            batch_ctx = (
                f"This is batch {batch_num} of {total_batches}. "
                f"Generate DIFFERENT values from all other batches — no repeated entries."
            )

        desc_line = f"Dataset: {description}" if description else f"Dataset: {dataset_name}"
        criteria_line = f"Extra criteria: {ai_criteria}" if ai_criteria else ""

        sample_ctx = ""
        if sample_rows:
            sample_ctx = "\nHere are some sample rows to match the style, formatting, and values:\n"
            for row in sample_rows[:5]:
                # Automatically format as CSV string
                sample_ctx += ", ".join(str(v) for v in row.values()) + "\n"

        prompt = f"""Generate exactly {rows} unique rows of synthetic data in CSV format with these columns:
{col_defs}
Column order (CSV): {col_names_str}

{desc_line}
{criteria_line}
{batch_ctx}
{sample_ctx}
Rules:
- Output ONLY comma-separated values. No header row. No markdown. No code blocks.
- Each row must have exactly {expected_cols} fields — strictly match the column count.
- Strings must NOT be wrapped in quotes unless the string contains a comma or quote.
- All {expected_cols} fields must be present on every row — never leave a field empty.
- Do NOT say "And so on...", "I've generated...", or add any trailing text.
- Each row must be unique and realistic.
- Respect nullable=false columns — they must always have a value.
- Respect any pattern constraints (e.g. integer ranges like 18-65).
"""
        return prompt.strip()

    @staticmethod
    def _render_schema(schema):
        """
        Render the schema into a readable prompt section.
        Supports both simple dicts {name, type} and rich dicts
        {name, type, nullable, pattern, distribution}.
        """
        lines = ["\nSchema Definition:"]
        lines.append(f"{'Column':<20} {'Type':<12} {'Nullable':<10} {'Pattern / Notes'}")
        lines.append("─" * 65)

        for col in schema:
            name = col.get("name", "")
            col_type = col.get("type", "string")
            nullable = "yes" if col.get("nullable", False) else "no"
            pattern = col.get("pattern", "")
            distribution = col.get("distribution", "")
            notes = " | ".join(filter(None, [pattern, distribution]))
            lines.append(f"  {name:<18} {col_type:<12} {nullable:<10} {notes}")

        lines.append("─" * 65)
        return "\n".join(lines)

    @staticmethod
    def _render_csv_columns(schema) -> str:
        """Render schema as a numbered column list for CSV-format prompts."""
        lines = []
        for i, col in enumerate(schema, 1):
            name     = col.get("name", "")
            col_type = col.get("type", "string")
            nullable = col.get("nullable", True)
            pattern  = col.get("pattern", "")
            note = f"  [range: {pattern}]" if pattern else ""
            req  = "" if nullable else "  [required, never empty]"
            lines.append(f"  {i}. {name} ({col_type}){note}{req}")
        return "\n".join(lines)

    @staticmethod
    def build_continuation(
        dataset_name,
        rows,
        description=None,
        schema=None,
        ai_criteria=None,
        last_row=None,
        batch_num=None,
        total_batches=None
    ):
        """
        Build a CSV-format continuation prompt.
        Used when the previous batch response was truncated mid-way.
        """
        col_defs = DatasetPromptBuilder._render_csv_columns(schema) if schema else ""
        expected_cols = len(schema) if schema else "?"
        col_names_str = ", ".join(c["name"] for c in schema) if schema else ""

        last_row_csv = ""
        if last_row and schema:
            last_row_csv = ", ".join(
                str(last_row.get(c["name"], "")) for c in schema
            )

        batch_ctx = ""
        if batch_num and total_batches:
            batch_ctx = f"This is a continuation of batch {batch_num} of {total_batches}."

        prompt = f"""Continue generating synthetic data for dataset '{dataset_name}'.
Generate exactly {rows} MORE rows in CSV format.

Columns:
{col_defs}
Column order: {col_names_str}

The last row already generated was:
{last_row_csv}

Continue from the NEXT row after this. Do NOT repeat that row.
{batch_ctx}

Rules:
- Output ONLY comma-separated values. No header. No markdown. No explanation.
- Each row must have exactly {expected_cols} fields.
- Strings must NOT be quoted unless they contain a comma or quote character.
- All fields must be present — never leave a field empty.
- Each row must be unique and realistic.
"""
        return prompt.strip()