from datetime import date

import pytest

from app import llm_client
from app.schemas import ExtractedClinicalData


class FakeResponse:
    def __init__(self, output: str):
        self.output = output

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": self.output}


def valid_json() -> str:
    return (
        '{"patient_name":"Fictional Patient","date_of_birth":"1980-01-02",'
        '"diagnosis_codes":["Z00.00"],"medications":[],"visit_date":"2026-01-03"}'
    )


def test_success_on_first_try(monkeypatch):
    responses = iter([FakeResponse(valid_json())])
    monkeypatch.setattr(llm_client.requests, "post", lambda *args, **kwargs: next(responses))

    result = llm_client.call_model("test-model", "Extract this note", ExtractedClinicalData)

    assert result.is_valid is True
    assert result.retry_count == 0
    assert result.data.patient_name == "Fictional Patient"


def test_success_after_one_retry(monkeypatch):
    responses = iter([FakeResponse('{"unexpected":"hallucination"}'), FakeResponse(valid_json())])
    monkeypatch.setattr(llm_client.requests, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)

    result = llm_client.call_model("test-model", "Extract this note", ExtractedClinicalData)

    assert result.is_valid is True
    assert result.retry_count == 1
    assert result.errors


def test_failure_after_all_retries(monkeypatch):
    responses = iter([FakeResponse('{"unexpected":1}'), FakeResponse('{"unexpected":2}'), FakeResponse('{"unexpected":3}')])
    monkeypatch.setattr(llm_client.requests, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)

    result = llm_client.call_model("test-model", "Extract this note", ExtractedClinicalData)

    assert result.is_valid is False
    assert result.data is None
    assert result.retry_count == 2
    assert len(result.errors) == 3