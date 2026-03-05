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
    primary_key: Optional[str] = Field(default=None, example="user_id", description="Explicitly set the primary key column for generation")
    sample_rows: Optional[List[dict]] = Field(default=None, description="Provide 3-5 example rows mapped exactly to your schema to give the LLM context")

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
                ],
                "primary_key": "user_id"
            }
        }
    }


class GenerateResponse(BaseModel):
    job_id: str
    dataset_name: str
    rows_generated: int
    columns: List[str]
    file_path: str          # local path (empty on App Runner)
    s3_key: Optional[str]   # S3 object key
    s3_url: Optional[str]   # presigned download URL (1 hour)
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


class AnalyzeResponse(BaseModel):
    dataset_name: str
    rows: int
    format: str
    description: Optional[str]
    ai_criteria: Optional[str]
    target_location: Optional[str]
    columns: Optional[List[dict]]
    sample_rows: Optional[List[dict]]
    primary_key: Optional[str]
    prompt_preview: str
    ready_to_generate: bool = True


# ─────────────────────────────────────────
# Helper: TSV save support
# ─────────────────────────────────────────

def _save_tsv(df, path):
    df.to_csv(path, index=False, sep="\t")


# ─────────────────────────────────────────
# Helper: Upload generated file to S3
# ─────────────────────────────────────────

def _upload_to_s3(file_path: str, dataset_name: str) -> tuple[str, str]:
    """
    Upload a local file to S3 and return (s3_key, presigned_url).
    The presigned URL is valid for 1 hour.
    """
    import boto3
    bucket = settings.s3_bucket
    filename = os.path.basename(file_path)
    s3_key = f"datasets/{dataset_name}/{filename}"

    s3 = boto3.client("s3", region_name=settings.aws_region)

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        logger.info(f"Creating S3 bucket: {bucket}")
        if settings.aws_region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": settings.aws_region}
            )

    # Upload the file
    s3.upload_file(file_path, bucket, s3_key)
    logger.info(f"Uploaded to s3://{bucket}/{s3_key}")

    # Generate presigned URL (1 hour expiry)
    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600
    )
    return s3_key, presigned_url


# ─────────────────────────────────────────
# Helper: Smart defaults for description / ai_criteria
# ─────────────────────────────────────────

def _build_defaults(dataset_name: str, columns: list | None) -> tuple[str, str]:
    """
    Generate a generic description and ai_criteria from the dataset name and columns
    when the user doesn't provide them. Works for any dataset.
    """
    readable = dataset_name.replace("_", " ").replace("-", " ").strip().title()

    col_names = [c["name"] if isinstance(c, dict) else str(c) for c in (columns or [])]
    col_summary = ", ".join(col_names[:6])
    if len(col_names) > 6:
        col_summary += f" and {len(col_names) - 6} more"

    description = (
        f"Realistic synthetic {readable} dataset"
        + (f" containing fields: {col_summary}." if col_summary else ".")
    )

    ai_criteria = (
        "Generate realistic, diverse, and internally consistent data. "
        "Values must reflect real-world distributions and relationships between columns. "
        "Avoid repetitive patterns, placeholder text, or sequential values. "
        "Ensure all fields have appropriate formats and plausible value ranges."
    )

    return description, ai_criteria


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


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Generation"])
def analyze_dataset(req: GenerateRequest):
    """
    Step 1 of 2 — Analyze & Preview before generating.

    Validates your request and returns a summary of:
    - All columns with name, type, nullable, pattern
    - The AUTO-DETECTED Primary Key (or the one you explicitly provided)
    - The exact prompt that will be sent to the LLM
    - Dataset settings (rows, format, etc.)

    Does NOT call the LLM. Once the user approves the primary_key, call POST /generate.
    """

    from core.input_processor import InputProcessor
    from prompt.dataset_prompt import DatasetPromptBuilder

    raw_input = {
        "dataset_name": req.dataset_name,
        "rows": str(req.rows),
        "format": req.format,
        "description": req.description,
        "ai_criteria": req.ai_criteria,
        "target_location": req.target_location,
        "schema": [col.model_dump(exclude_none=True) for col in req.columns] if req.columns else None,
        "schema_file": None,
        "sample_rows": req.sample_rows,
        "sample_file": None,
    }

    processor = InputProcessor()
    request = processor.build_request(raw_input)

    # Auto-detect PK if not explicitly provided
    detected_pk = req.primary_key
    if not detected_pk and request.get("schema"):
        from prompt.dataset_prompt import _detect_primary_key
        detected_pk = _detect_primary_key(request["schema"])

    # Fill in smart defaults if user didn't provide description / ai_criteria
    _desc, _criteria = _build_defaults(req.dataset_name, request.get("schema"))
    effective_description = request.get("description") or _desc
    effective_criteria    = request.get("ai_criteria")  or _criteria

    prompt = DatasetPromptBuilder.build(
        dataset_name=request["dataset_name"],
        rows=request["rows"],
        description=effective_description,
        schema=request.get("schema"),
        ai_criteria=effective_criteria,
        primary_key=detected_pk
    )

    return AnalyzeResponse(
        dataset_name=req.dataset_name,
        rows=req.rows,
        format=req.format,
        description=req.description,
        ai_criteria=req.ai_criteria,
        target_location=req.target_location,
        columns=request.get("schema"),
        sample_rows=request.get("sample_rows"),
        primary_key=detected_pk,
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

    # Build internal request dict — fill in smart defaults if not provided
    _desc, _criteria = _build_defaults(
        req.dataset_name,
        [col.model_dump(exclude_none=True) for col in req.columns] if req.columns else None
    )
    raw_input = {
        "dataset_name": req.dataset_name,
        "rows": str(req.rows),
        "format": req.format,
        "description": req.description or _desc,
        "ai_criteria": req.ai_criteria or _criteria,
        "target_location": req.target_location,
        "schema": [col.model_dump(exclude_none=True) for col in req.columns] if req.columns else None,
        "primary_key": req.primary_key,
        "schema_file": None,
        "sample_rows": req.sample_rows,
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
        import pandas as pd
        tsv_path = file_path.replace(".csv", ".tsv")
        df = pd.read_csv(file_path)
        df.to_csv(tsv_path, index=False, sep="\t")
        os.remove(file_path)
        file_path = tsv_path

    job_id = result["job_id"]
    generated_at = result["generated_at"]

    # ── Upload to S3 ──────────────────────────────────────────────
    s3_key = None
    s3_url = None
    try:
        s3_key, s3_url = _upload_to_s3(file_path, req.dataset_name)
        # Clean up local file after successful S3 upload
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted local file after S3 upload: {file_path}")
    except Exception as e:
        logger.warning(f"S3 upload failed, keeping local file: {e}")

    # ── Save to DynamoDB with s3_key ──────────────────────────────
    try:
        db.save_job({
            "job_id":       job_id,
            "dataset_name": result["dataset_name"],
            "rows":         result["rows_generated"],
            "format":       req.format,
            "columns":      result["columns"],
            "file_path":    s3_key or file_path,
            "s3_key":       s3_key,
            "generated_at": generated_at,
            "status":       "success"
        })
    except Exception as e:
        logger.warning(f"Failed to save job to DynamoDB: {e}")

    response = GenerateResponse(
        job_id=job_id,
        dataset_name=result["dataset_name"],
        rows_generated=result["rows_generated"],
        columns=result["columns"],
        file_path=s3_key or file_path,
        s3_key=s3_key,
        s3_url=s3_url,
        format=req.format,
        generated_at=generated_at
    )

    logger.info(f"Generation complete: s3://{settings.s3_bucket}/{s3_key}")
    return response


@app.get("/history", response_model=List[HistoryItem], tags=["Generation"])
def get_history():
    """Return all past generation runs from DynamoDB (persists across restarts)"""
    try:
        return db.get_all_jobs()
    except Exception as e:
        logger.error(f"DynamoDB scan failed: {e}")
        raise HTTPException(status_code=503, detail=f"History unavailable: {e}")


@app.get("/download/{job_id}", tags=["Generation"])
def download_file(job_id: str):
    """
    Get a presigned S3 download URL for a generated dataset.

    Frontend redirects the user to the returned `download_url` to trigger
    the browser download directly from S3.
    """
    import boto3

    job = None
    try:
        job = db.get_job(job_id)
    except Exception as e:
        logger.error(f"DynamoDB get_job failed: {e}")
        raise HTTPException(status_code=503, detail=f"Could not look up job: {e}")

    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    # Prefer s3_key stored in DynamoDB
    s3_key = job.get("s3_key") or job.get("file_path")

    if not s3_key or not s3_key.startswith("datasets/"):
        raise HTTPException(
            status_code=404,
            detail=f"No S3 file found for job '{job_id}'. Legacy local-only jobs cannot be downloaded."
        )

    try:
        s3 = boto3.client("s3", region_name=settings.aws_region)
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": s3_key},
            ExpiresIn=3600
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise HTTPException(status_code=500, detail=f"Could not generate download URL: {e}")

    return {
        "job_id": job_id,
        "download_url": presigned_url,
        "s3_key": s3_key,
        "expires_in_seconds": 3600
    }


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


@app.get("/schema/sample", tags=["Schema"])
def get_sample_schema():
    """Returns a sample schema JSON you can use as a template"""
    return [
        {"name": "id",          "type": "uuid",    "nullable": False},
        {"name": "full_name",   "type": "name",    "nullable": False},
        {"name": "email",       "type": "email",   "nullable": False},
        {"name": "age",         "type": "integer", "nullable": False, "pattern": "18-65"},
        {"name": "city",        "type": "string",  "nullable": True},
        {"name": "signup_date", "type": "date",    "nullable": False,
         "pattern": "2020-01-01 to 2025-12-31"},
        {"name": "is_active",   "type": "boolean", "nullable": False}
    ]


@app.get("/schema/types", tags=["Schema"])
def get_supported_types():
    """Lists all supported column types, distributions, and output formats"""
    return {
        "types": [
            "string", "integer", "float", "boolean",
            "date", "datetime", "email", "phone",
            "uuid", "name", "address", "url"
        ],
        "distributions": ["uniform", "normal", "skewed", "random"],
        "formats": ["csv", "json", "parquet", "tsv"]
    }

