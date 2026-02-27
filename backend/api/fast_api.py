"""
FastAPI Application for Data Synthesizer
Exposes the dataset generation pipeline via HTTP endpoints
"""

import sys
import os
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__ + "/..")))

from core.data_synthesizer import DataSynthesizer
from core.input_processor import InputProcessor
from config.settings import get_settings
from logger.logger import get_logger
from db.dynamo_history import DynamoHistory

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────

settings = get_settings()
logger = get_logger(__name__)

# ─────────────────────────────────────────
# DynamoDB history store
# ─────────────────────────────────────────
db = DynamoHistory()


@asynccontextmanager
async def lifespan(app):
    """Run on startup: ensure DynamoDB table exists"""
    try:
        DynamoHistory.create_table_if_not_exists()
        logger.info(f"DynamoDB history table ready: {settings.dynamo_history_table}")
    except Exception as e:
        logger.warning(f"DynamoDB table setup failed (history may not persist): {e}")
    yield


app = FastAPI(
    title="Data Synthesizer API",
    description="Generate synthetic datasets using AWS Bedrock AI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────

class SchemaColumn(BaseModel):
    name: str
    type: str = "string"
    nullable: bool = False
    pattern: Optional[str] = None
    distribution: Optional[str] = None   # e.g. uniform, normal

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "email",
                "type": "email",
                "nullable": False,
                "pattern": None,
                "distribution": "uniform"
            }
        }
    }


class GenerateRequest(BaseModel):
    dataset_name: str = Field(..., example="users_dataset")
    rows: int = Field(..., gt=0, le=10_000_000, example=10)
    format: str = Field(default="csv", pattern="^(csv|json|parquet|tsv)$")
    description: Optional[str] = Field(default=None, example="User profile dataset for testing")
    ai_criteria: Optional[str] = Field(default=None, example="Generate realistic US-based users aged 18-65")
    target_location: Optional[str] = Field(default=None, example=None)
    columns: Optional[List[SchemaColumn]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_name": "users_dataset",
                "rows": 10,
                "format": "csv",
                "description": "User profile dataset for testing",
                "ai_criteria": "Generate realistic US-based users aged 18-65",
                "target_location": None,
                "columns": [
                    {"name": "user_id",    "type": "uuid",    "nullable": False},
                    {"name": "full_name",  "type": "name",    "nullable": False},
                    {"name": "email",      "type": "email",   "nullable": False},
                    {"name": "age",        "type": "integer", "nullable": False, "pattern": "18-65"},
                    {"name": "city",       "type": "string",  "nullable": True},
                    {"name": "is_active",  "type": "boolean", "nullable": False}
                ]
            }
        }
    }


class GenerateResponse(BaseModel):
    job_id: str
    dataset_name: str
    rows_generated: int
    columns: List[str]
    file_path: str
    format: str
    generated_at: str


class HistoryItem(BaseModel):
    job_id: str
    dataset_name: str
    rows: int
    format: str
    columns: List[str]
    file_path: str
    generated_at: str
    status: str


class PreviewResponse(BaseModel):
    dataset_name: str
    rows: int
    format: str
    mode: str
    synthesizer: str
    description: Optional[str]
    ai_criteria: Optional[str]
    target_location: Optional[str]
    columns: Optional[List[dict]]
    prompt_preview: str
    ready_to_generate: bool = True


# ─────────────────────────────────────────
# Helper: tsv save support
# ─────────────────────────────────────────

def _save_tsv(df, path):
    df.to_csv(path, index=False, sep="\t")


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Health check"""
    return {
        "status": "ok",
        "model": settings.bedrock_model,
        "environment": settings.environment
    }


@app.post("/preview", response_model=PreviewResponse, tags=["Generation"])
def preview_dataset(req: GenerateRequest):
    """
    Step 1 of 2 — Preview before generating.

    Validates your request and returns a summary of:
    - All columns with name, type, nullable, pattern
    - The exact prompt that will be sent to the LLM
    - Dataset settings (rows, format, etc.)

    Does NOT call the LLM. Call POST /generate to actually produce the data.
    """

    from core.input_processor import InputProcessor
    from prompt.dataset_prompt import DatasetPromptBuilder
    from core.schema_resolver import SchemaResolver
    from core.data_synthesizer import DataSynthesizer

    raw_input = {
        "dataset_name": req.dataset_name,
        "rows": str(req.rows),
        "format": req.format,
        "mode": req.mode,
        "synthesizer": req.synthesizer,
        "description": req.description,
        "ai_criteria": req.ai_criteria,
        "target_location": req.target_location,
        "schema": [col.model_dump(exclude_none=True) for col in req.columns] if req.columns else None,
        "schema_file": None,
        "sample_file": None,
    }

    processor = InputProcessor()
    request = processor.build_request(raw_input)

    resolved_schema = SchemaResolver.resolve(schema=request.get("schema"))
    if not resolved_schema and request.get("description"):
        synth = DataSynthesizer()
        resolved_schema = synth._discover_schema(
            dataset_name=request["dataset_name"],
            description=request["description"],
            ai_criteria=request.get("ai_criteria")
        )

    prompt = DatasetPromptBuilder.build(
        dataset_name=request["dataset_name"],
        rows=request["rows"],
        description=request.get("description"),
        schema=resolved_schema,
        ai_criteria=request.get("ai_criteria"),
    )

    return PreviewResponse(
        dataset_name=req.dataset_name,
        rows=req.rows,
        format=req.format,
        mode=req.mode,
        synthesizer=req.synthesizer,
        description=req.description,
        ai_criteria=req.ai_criteria,
        target_location=req.target_location,
        columns=resolved_schema,
        prompt_preview=prompt,
        ready_to_generate=True
    )



@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
def generate_dataset(req: GenerateRequest):
    """
    Generate a synthetic dataset.
    - Pass `schema` array to define columns explicitly.
    - Leave `schema` null to let AI decide the structure from `description`.
    """

    logger.info(f"Received generate request: {req.dataset_name} | {req.rows} rows | {req.format}")

    # Build internal request dict
    raw_input = {
        "dataset_name": req.dataset_name,
        "rows": str(req.rows),
        "format": req.format,
        "description": req.description,
        "ai_criteria": req.ai_criteria,
        "target_location": req.target_location,
        "schema": [col.model_dump(exclude_none=True) for col in req.columns] if req.columns else None,
        "schema_file": None,
        "sample_file": None,
    }

    processor = InputProcessor()
    request = processor.build_request(raw_input)

    # TSV: synthesizer saves as csv by default, we handle it separately
    save_fmt = "csv" if req.format == "tsv" else req.format
    request["format"] = save_fmt

    try:
        synth = DataSynthesizer()
        result = synth.generate(request)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # If TSV, rename file extension
    file_path = result["file_path"]
    if req.format == "tsv":
        tsv_path = file_path.replace(".csv", ".tsv")
        import pandas as pd
        import os
        df = pd.read_csv(file_path)
        df.to_csv(tsv_path, index=False, sep="\t")
        os.remove(file_path)
        file_path = tsv_path

    job_id = result["job_id"]
    generated_at = result["generated_at"]

    response = GenerateResponse(
        job_id=job_id,
        dataset_name=result["dataset_name"],
        rows_generated=result["rows_generated"],
        columns=result["columns"],
        file_path=file_path,
        format=req.format,
        generated_at=generated_at
    )

    logger.info(f"Generation complete: {file_path}")
    return response


@app.get("/download/{job_id}", tags=["Generation"])
def download_file(job_id: str):
    """
    Download the generated dataset file by job_id.

    Frontend calls this after POST /generate to let the user download the file.
    Returns the file as a browser-downloadable attachment.
    """
    job = None
    try:
        job = db.get_job(job_id)
    except Exception as e:
        logger.error(f"DynamoDB get_job failed: {e}")
        raise HTTPException(status_code=503, detail=f"Could not look up job: {e}")

    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    file_path = job["file_path"]

    # Resolve relative path from backend directory
    if not os.path.isabs(file_path):
        backend_dir = os.path.dirname(os.path.abspath(__file__ + "/.."))
        file_path = os.path.normpath(os.path.join(backend_dir, file_path.lstrip("./")))

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    fmt = job.get("format", "csv")
    media_types = {
        "csv":     "text/csv",
        "json":    "application/json",
        "parquet": "application/octet-stream",
        "tsv":     "text/tab-separated-values"
    }

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type=media_types.get(fmt, "application/octet-stream")
    )


@app.post("/schema/upload", tags=["Schema"])
async def upload_schema_file(file: UploadFile = File(...)):
    """
    Upload a schema file and get back columns + sample rows.

    Supported formats:
      - .json  → reads columns array directly
      - .csv   → auto-infers schema from column names/types + extracts 5 sample rows

    Returns:
      {
        filename, file_type, columns_count,
        columns: [...],       ← pass this to POST /generate as 'columns'
        sample_rows: [...]    ← optional, only for CSV uploads
      }
    """
    import io
    import pandas as pd

    filename = file.filename or ""
    contents = await file.read()

    # ─── JSON Schema File ───────────────────────────────────────
    if filename.endswith(".json"):
        try:
            schema = json.loads(contents.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(400, detail=f"Invalid JSON: {e}")

        if not isinstance(schema, list):
            raise HTTPException(400, detail="JSON schema must be an array of column objects")

        for i, col in enumerate(schema):
            if not isinstance(col, dict) or "name" not in col:
                raise HTTPException(400, detail=f"Column {i} must be an object with a 'name' field")

        return {
            "filename": filename,
            "file_type": "json",
            "columns_count": len(schema),
            "columns": schema,
            "sample_rows": None
        }

    # ─── CSV File → Extract Schema + Sample Rows ────────────────
    elif filename.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(400, detail=f"Could not read CSV: {e}")

        if df.empty:
            raise HTTPException(400, detail="CSV file is empty")

        # Dtype → column type mapping
        dtype_map = {
            "object":          "string",
            "int64":           "integer",
            "int32":           "integer",
            "float64":         "float",
            "float32":         "float",
            "bool":            "boolean",
            "datetime64[ns]":  "datetime",
        }

        columns = []
        for col_name, dtype in df.dtypes.items():
            col_type = dtype_map.get(str(dtype), "string")

            # Detect email-like columns by name
            name_lower = col_name.lower()
            if "email" in name_lower:
                col_type = "email"
            elif "phone" in name_lower or "mobile" in name_lower:
                col_type = "phone"
            elif "date" in name_lower or "time" in name_lower:
                col_type = "date"
            elif "uuid" in name_lower or "id" == name_lower:
                col_type = "uuid"
            elif "url" in name_lower or "link" in name_lower:
                col_type = "url"

            # Detect nullable: if any value is null in CSV
            has_nulls = df[col_name].isnull().any()

            columns.append({
                "name": col_name,
                "type": col_type,
                "nullable": bool(has_nulls)
            })

        # Extract up to 5 sample rows
        sample_rows = df.head(5).fillna("").to_dict(orient="records")

        return {
            "filename": filename,
            "file_type": "csv",
            "total_rows_in_file": len(df),
            "columns_count": len(columns),
            "columns": columns,
            "sample_rows": sample_rows
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{filename}'. Upload a .json or .csv file."
        )

