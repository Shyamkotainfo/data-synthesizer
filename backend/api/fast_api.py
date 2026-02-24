"""
FastAPI Application for Data Synthesizer
Exposes the dataset generation pipeline via HTTP endpoints
"""

import sys
import os
import json
import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__ + "/..")))

from core.data_synthesizer import DataSynthesizer
from core.input_processor import InputProcessor
from config.settings import get_settings
from logger.logger import get_logger

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title="Data Synthesizer API",
    description="Generate synthetic datasets using AWS Bedrock AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory generation history
_history: List[dict] = []


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

    prompt = DatasetPromptBuilder.build(
        dataset_name=request["dataset_name"],
        rows=request["rows"],
        description=request.get("description"),
        schema=request.get("schema"),
        ai_criteria=request.get("ai_criteria"),
    )

    return PreviewResponse(
        dataset_name=req.dataset_name,
        rows=req.rows,
        format=req.format,
        description=req.description,
        ai_criteria=req.ai_criteria,
        target_location=req.target_location,
        columns=request.get("schema"),
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

    job_id = str(uuid.uuid4())
    generated_at = datetime.now().isoformat()

    response = GenerateResponse(
        job_id=job_id,
        dataset_name=result["dataset_name"],
        rows_generated=result["rows_generated"],
        columns=result["columns"],
        file_path=file_path,
        format=req.format,
        generated_at=generated_at
    )

    # Save to history
    _history.append({
        "job_id": job_id,
        "dataset_name": result["dataset_name"],
        "rows": result["rows_generated"],
        "format": req.format,
        "columns": result["columns"],
        "file_path": file_path,
        "generated_at": generated_at,
        "status": "success"
    })

    logger.info(f"Generation complete: {file_path}")
    return response


@app.get("/history", response_model=List[HistoryItem], tags=["Generation"])
def get_history():
    """Return all past generation runs (in-memory, resets on restart)"""
    return list(reversed(_history))


@app.get("/schema/sample", tags=["Schema"])
def get_sample_schema():
    """Returns a sample schema JSON you can use as a template"""
    return [
        {"name": "id",         "type": "uuid",    "nullable": False},
        {"name": "full_name",  "type": "name",    "nullable": False},
        {"name": "email",      "type": "email",   "nullable": False},
        {"name": "age",        "type": "integer", "nullable": False, "pattern": "18-65"},
        {"name": "city",       "type": "string",  "nullable": True},
        {"name": "signup_date","type": "date",    "nullable": False,
         "pattern": "2020-01-01 to 2025-12-31"},
        {"name": "is_active",  "type": "boolean", "nullable": False}
    ]


@app.get("/schema/types", tags=["Schema"])
def get_supported_types():
    """Lists all supported column types"""
    return {
        "types": [
            "string", "integer", "float", "boolean",
            "date", "datetime", "email", "phone",
            "uuid", "name", "address", "url"
        ],
        "distributions": ["uniform", "normal", "skewed", "random"]
    }
