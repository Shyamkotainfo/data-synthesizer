import json
import boto3
from typing import Optional
from config.settings import settings
from logger.logger import get_logger

logger = get_logger(__name__)


def get_llm_client(region: Optional[str] = None):
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region or settings.aws_region
    )


def invoke_llm(
    prompt: str,
    model_id: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> str:

    model_id = model_id or settings.bedrock_model
    client = get_llm_client()

    logger.debug(f"Invoking Bedrock model: {model_id}")

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "topP": 0.9
            }
        })
    )

    payload = json.loads(response["body"].read())
    return payload["output"]["message"]["content"][0]["text"]


class LLMQuery:
    def __init__(self):
        self.model_id = settings.bedrock_model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        return invoke_llm(
            prompt=prompt,
            model_id=self.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )