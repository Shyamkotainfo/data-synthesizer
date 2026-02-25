"""
DynamoDB-backed generation history store.

Table schema:
  PK: job_id (String)
  Attributes: dataset_name, rows, format, columns (JSON list), file_path,
              generated_at (ISO string), status, ttl (optional, epoch seconds)
"""

import json
import time
import boto3
from botocore.exceptions import ClientError

from config.settings import get_settings
from logger.logger import get_logger

logger = get_logger(__name__)


class DynamoHistory:
    """Read/write generation history to DynamoDB."""

    def __init__(self):
        self.settings = get_settings()
        self.table_name = self.settings.dynamo_history_table
        self._table = None

    # ─────────────────────────────────────────
    # Lazy table accessor
    # ─────────────────────────────────────────
    @property
    def table(self):
        if self._table is None:
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=self.settings.aws_region
            )
            self._table = dynamodb.Table(self.table_name)
        return self._table

    # ─────────────────────────────────────────
    # Write a new job record
    # ─────────────────────────────────────────
    def save_job(self, job: dict):
        """
        Save a generation job to DynamoDB.
        job must contain: job_id, dataset_name, rows, format,
                          columns, file_path, generated_at, status
        """
        item = {
            "job_id":       job["job_id"],
            "dataset_name": job["dataset_name"],
            "rows":         job["rows"],
            "format":       job["format"],
            "columns":      json.dumps(job["columns"]),   # store list as JSON string
            "file_path":    job["file_path"],
            "generated_at": job["generated_at"],
            "status":       job.get("status", "success"),
            # Optional TTL — keeps table clean; 90 days from now
            "ttl":          int(time.time()) + 90 * 24 * 3600
        }
        try:
            self.table.put_item(Item=item)
            logger.info(f"Saved job {job['job_id']} to DynamoDB table '{self.table_name}'")
        except ClientError as e:
            logger.error(f"DynamoDB put_item failed: {e}")
            raise

    # ─────────────────────────────────────────
    # Read a single job by job_id
    # ─────────────────────────────────────────
    def get_job(self, job_id: str) -> dict | None:
        """Return a single job record, or None if not found."""
        try:
            resp = self.table.get_item(Key={"job_id": job_id})
            item = resp.get("Item")
            if item:
                item["columns"] = json.loads(item["columns"])
            return item
        except ClientError as e:
            logger.error(f"DynamoDB get_item failed: {e}")
            raise

    # ─────────────────────────────────────────
    # Read all jobs (scan — suitable for small tables)
    # ─────────────────────────────────────────
    def get_all_jobs(self) -> list[dict]:
        """
        Return all jobs sorted by generated_at descending.
        Uses a full table scan — acceptable for history tables.
        """
        try:
            resp = self.table.scan()
            items = resp.get("Items", [])

            # Handle DynamoDB pagination
            while "LastEvaluatedKey" in resp:
                resp = self.table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
                items.extend(resp.get("Items", []))

            # Deserialize columns JSON string
            for item in items:
                if isinstance(item.get("columns"), str):
                    item["columns"] = json.loads(item["columns"])
                # Remove internal TTL from API response
                item.pop("ttl", None)

            # Sort newest first
            items.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
            return items

        except ClientError as e:
            logger.error(f"DynamoDB scan failed: {e}")
            raise

    # ─────────────────────────────────────────
    # Create the table if it doesn't exist
    # ─────────────────────────────────────────
    @classmethod
    def create_table_if_not_exists(cls):
        """
        One-time setup: create the DynamoDB table.
        Safe to call on every startup — no-ops if table already exists.
        """
        settings = get_settings()
        table_name = settings.dynamo_history_table

        dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)

        existing = [t.name for t in dynamodb.tables.all()]
        if table_name in existing:
            logger.info(f"DynamoDB table '{table_name}' already exists")
            return

        logger.info(f"Creating DynamoDB table '{table_name}'...")
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "job_id", "KeyType": "HASH"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "job_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST"   # on-demand pricing, no capacity planning
        )
        table.wait_until_exists()
        logger.info(f"Table '{table_name}' created successfully ✅")
