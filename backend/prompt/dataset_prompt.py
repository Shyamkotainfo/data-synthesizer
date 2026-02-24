import json


class DatasetPromptBuilder:

    @staticmethod
    def build(
        dataset_name,
        rows,
        description=None,
        schema=None,
        sample_rows=None,
        ai_criteria=None
    ):

        sections = []

        sections.append(f"Generate {rows} rows of realistic tabular data.")
        sections.append(f"Dataset Name: {dataset_name}")

        if description:
            sections.append(f"\nDescription:\n{description}")

        if schema:
            sections.append(DatasetPromptBuilder._render_schema(schema))

        if sample_rows:
            sections.append(
                f"\nSample Rows:\n{json.dumps(sample_rows, indent=2)}"
            )

        if ai_criteria:
            sections.append(f"\nAI Criteria:\n{ai_criteria}")

        sections.append("""
Rules:
- Return ONLY valid JSON
- Return a JSON array of objects
- No explanation
- No markdown fences
- Follow schema strictly if provided
- Respect nullable rules: nullable=false means no null values
- Respect pattern constraints when specified
""")

        return "\n".join(sections)

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