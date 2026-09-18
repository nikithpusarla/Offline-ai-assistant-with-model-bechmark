import os
from datetime import date
from pathlib import Path

import requests
import yaml
from fastapi import FastAPI, HTTPException

from app.audit_log import log_extraction
from app.llm_client import call_model
from app.redaction import redact
from app.schemas import ClinicalNoteRequest, ExtractedClinicalData, RedactedClinicalData


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "models_config.yaml").read_text())
DEFAULT_MODEL = os.getenv("MODEL_NAME") or CONFIG.get("production_model") or next(item["name"] for item in CONFIG["models"] if item.get("recommended", False))
app = FastAPI(title="Local Offline Clinical Note Assistant")


def _restore(value, mapping: list[dict]):
    if isinstance(value, str):
        for item in mapping:
            value = value.replace(item["placeholder"], item["original_value"])
        return value
    if isinstance(value, list):
        return [_restore(item, mapping) for item in value]
    return value


@app.get("/health")
def health() -> dict:
    try:
        response = requests.get(f"{CONFIG['ollama_base_url']}/api/tags", timeout=3)
        response.raise_for_status()
        ollama = "reachable"
    except requests.RequestException:
        ollama = "unreachable"
    return {"status": "ok" if ollama == "reachable" else "degraded", "ollama": ollama, "external_network": False}


@app.post("/extract", response_model=ExtractedClinicalData)
def extract(request: ClinicalNoteRequest) -> ExtractedClinicalData:
    redacted_text, mapping = redact(request.raw_text)
    result = call_model(DEFAULT_MODEL, redacted_text, RedactedClinicalData)
    if not result.is_valid or result.data is None:
        log_extraction(DEFAULT_MODEL, mapping, result.retry_count, False)
        raise HTTPException(status_code=422, detail={"errors": result.errors, "retry_count": result.retry_count})
    restored = _restore(result.data.model_dump(), mapping)
    try:
        validated = ExtractedClinicalData.model_validate({
            **restored,
            "date_of_birth": date.fromisoformat(restored["date_of_birth"]) if restored.get("date_of_birth") else None,
            "visit_date": date.fromisoformat(restored["visit_date"]) if restored.get("visit_date") else None,
        })
    except (ValueError, TypeError):
        log_extraction(DEFAULT_MODEL, mapping, result.retry_count, False)
        raise HTTPException(status_code=422, detail="Restored model output failed final clinical schema validation")
    log_extraction(DEFAULT_MODEL, mapping, result.retry_count, True)
    return validated