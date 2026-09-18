import pytest

from app import llm_client
from app.schemas import RedactedClinicalData


def test_non_local_ollama_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://example.invalid")
    with pytest.raises(ValueError, match="localhost"):
        llm_client._ollama_url()


def test_client_only_constructs_local_endpoint(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert llm_client._ollama_url() == "http://localhost:11434"