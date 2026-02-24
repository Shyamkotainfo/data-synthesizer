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
            sections.append(
                f"\nSchema Definition:\n{json.dumps(schema, indent=2)}"
            )

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
- No markdown
- Follow schema strictly if provided
""")

        return "\n".join(sections)