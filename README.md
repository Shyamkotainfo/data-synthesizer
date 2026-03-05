# AI Data Synthesizer

Generate realistic synthetic datasets using AWS Bedrock LLMs — via a FastAPI REST API or an interactive CLI.

---

## Features

- **Upload a CSV or JSON** to extract schema + sample rows automatically
- **AI-powered generation** using Amazon Nova (Bedrock) — realistic, diverse data
- **Smart defaults** — description and AI criteria are auto-generated from column names when not provided
- **Parallel batch generation** with primary key uniqueness enforcement
- **S3 storage** — generated files are uploaded to S3 and served via presigned download URLs
- **DynamoDB history** — every generation job is persisted with a unique `job_id`
- **Docker + GitHub Actions CI/CD** — auto-deploys to AWS App Runner on push to `dev`, `staging`, or `prod`

---

## API Workflow

### Step 1 — Upload Schema (optional)
```http
POST /schema/upload
Content-Type: multipart/form-data
Body: file = <your CSV or JSON file>
```
**Returns:** `columns[]`, `sample_rows[]`, `total_rows_in_file`

Use this to extract column definitions + sample data from your existing dataset.

---

### Step 2 — Analyze (Preview)
```http
POST /analyze
Content-Type: application/json
```
```json
{
  "dataset_name": "police_stops",
  "rows": 500,
  "format": "csv",
  "columns": [ { "name": "stop_date", "type": "date", "nullable": false }, ... ],
  "sample_rows": [ { "stop_date": "2005-01-02", ... }, ... ]
}
```
**Returns:** auto-detected `primary_key`, `prompt_preview`, `columns`, `sample_rows`

---

### Step 3 — Generate
```http
POST /generate
Content-Type: application/json
```
```json
{
  "dataset_name": "police_stops",
  "rows": 500,
  "format": "csv",
  "primary_key": "stop_date",
  "columns": [ ... ],
  "sample_rows": [ ... ]
}
```
**Returns:**
```json
{
  "job_id": "abc-123",
  "rows_generated": 500,
  "s3_key": "datasets/police_stops/police_stops_20260304.csv",
  "s3_url": "https://s3.amazonaws.com/...?Expires=...",
  "generated_at": "2026-03-04T12:00:00"
}
```

> `s3_url` is a **presigned URL** valid for 1 hour. Pass it directly to the browser to trigger a download.

---

### Step 4 — Download
```http
GET /download/{job_id}
```
**Returns:**
```json
{
  "download_url": "https://s3.amazonaws.com/...?Expires=...",
  "expires_in_seconds": 3600
}
```

---

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/schema/types` | GET | List supported column types |
| `/schema/sample` | GET | Get a sample schema JSON |
| `/history` | GET | All past generation jobs from DynamoDB |

---

## Running Locally

```bash
cd backend
pip install -r requirements.txt

# Start API server
python main.py --mode api    # → http://localhost:8081/docs

# Start CLI
python main.py               # → interactive mode
```

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `dev` | `development` / `staging` / `production` (or aliases `dev`, `prod`) |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock, S3, DynamoDB |
| `BEDROCK_MODEL` | `amazon.nova-pro-v1:0` | Bedrock model ID |
| `S3_BUCKET` | `data-synthesizer-output` | S3 bucket for generated files |
| `DYNAMO_HISTORY_TABLE` | `data_synthesizer_history` | DynamoDB table for job history |
| `OUTPUT_DIR` | `./data/output` | Local output directory (CLI mode) |
| `MAX_ROWS_PER_REQUEST` | `5000` | Row limit per API request |

---

## Deployment (GitHub Actions → AWS App Runner)

The workflow in `.github/workflows/deploy-backend.yml` runs automatically when you push to `dev`, `staging`, or `prod` and any file in `backend/` changes.

### What it does:
1. Builds a Docker image (`linux/amd64`) from `backend/`
2. Pushes to ECR (`data-synthesizer-backend:<branch>`)
3. Auto-creates `AppRunnerECRAccessRole` (ECR pull access) if it doesn't exist
4. Auto-creates `AppRunnerInstanceRole` (Bedrock + S3 + DynamoDB access) if it doesn't exist
5. Creates or updates `data-synthesizer-<branch>` App Runner service

### Required GitHub Secrets:

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

### Branch → Environment mapping:

| Branch | `ENVIRONMENT` value | App Runner service |
|---|---|---|
| `dev` | `development` | `data-synthesizer-dev` |
| `staging` | `staging` | `data-synthesizer-staging` |
| `prod` | `production` | `data-synthesizer-prod` |

---

## Docker

```bash
# Build locally (from repo root)
docker build -f backend/Dockerfile -t data-synthesizer .

# Run
docker run -p 8080:8080 \
  -e AWS_REGION=us-east-1 \
  -e S3_BUCKET=data-synthesizer-output \
  -e BEDROCK_MODEL=amazon.nova-pro-v1:0 \
  -e DYNAMO_HISTORY_TABLE=data_synthesizer_history \
  -e ENVIRONMENT=development \
  data-synthesizer
```
