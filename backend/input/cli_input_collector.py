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

    NOTE: input() prompts are commented out for debugging.
    Dummy values are used instead.
    """

    print("\n--- Dataset Configuration (DEBUG MODE - using dummy values) ---")

    # --- DUMMY DEBUG VALUES ---
    dataset_name = "users_debug"
    rows = "10"
    fmt = "csv"
    description = "A dataset of user profiles with name, age, email, and city"
    schema_file = ""
    sample_file = ""
    ai_criteria = "Generate realistic US-based user data"
    target_location = ""

    # --- ORIGINAL INPUT PROMPTS (commented out for debugging) ---
    # dataset_name = input("Dataset Name: ").strip()
    # rows = input("Number of Rows: ").strip()
    # fmt = input("Format (csv/parquet/json): ").strip().lower()
    # print("\n--- Optional Fields ---")
    # description = input("Description (optional): ").strip()
    # schema_file = input("Schema File Path (optional): ").strip()
    # sample_file = input("Sample File Path (optional): ").strip()
    # ai_criteria = input("AI Criteria (optional): ").strip()
    # target_location = input("Target Location (optional): ").strip()

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