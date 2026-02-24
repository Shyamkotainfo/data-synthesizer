"""
CLI Input Collector
Responsible ONLY for collecting raw input from CLI.
No validation.
No business logic.
No file parsing.
"""

def collect_generation_input():
    """
    Collects raw user input.
    Returns raw request dictionary.
    """

    print("\n--- Dataset Configuration ---")

    dataset_name = input("Dataset Name: ").strip()
    rows = input("Number of Rows: ").strip()
    fmt = input("Format (csv/parquet/json): ").strip().lower()

    print("\n--- Optional Fields ---")

    description = input("Description (optional): ").strip()
    schema_file = input("Schema File Path (optional): ").strip()
    sample_file = input("Sample File Path (optional): ").strip()
    ai_criteria = input("AI Criteria (optional): ").strip()
    target_location = input("Target Location (optional): ").strip()

    return {
        "dataset_name": dataset_name or None,
        "rows": rows or None,
        "format": fmt or None,
        "description": description or None,
        "schema_file": schema_file or None,
        "sample_file": sample_file or None,
        "ai_criteria": ai_criteria or None,
        "target_location": target_location or None
    }