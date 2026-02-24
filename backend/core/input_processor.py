import json
import pandas as pd


class InputProcessor:
    """
    Handles normalization of raw user input
    Converts CLI input into standardized request object
    """

    def build_request(self, raw_input: dict) -> dict:

        request = {
            "dataset_name": raw_input.get("dataset_name"),
            "rows": int(raw_input.get("rows")),
            "description": raw_input.get("description"),
            "format": raw_input.get("format"),
            "ai_criteria": raw_input.get("ai_criteria"),
            "target_location": raw_input.get("target_location"),
            "schema": None,
            "sample_rows": None
        }

        # Handle schema file
        if raw_input.get("schema_file"):
            with open(raw_input["schema_file"], "r") as f:
                request["schema"] = json.load(f)

        # Handle manual schema
        if raw_input.get("schema"):
            request["schema"] = raw_input["schema"]

        # Handle sample file
        if raw_input.get("sample_file"):
            request["sample_rows"] = self._load_sample_file(
                raw_input["sample_file"]
            )

        # Handle pasted sample rows
        if raw_input.get("sample_rows"):
            request["sample_rows"] = raw_input["sample_rows"]

        return request

    def _load_sample_file(self, path):

        if path.endswith(".csv"):
            df = pd.read_csv(path)
            return df.head(5).to_dict(orient="records")

        elif path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)

        else:
            raise ValueError("Unsupported sample file format")