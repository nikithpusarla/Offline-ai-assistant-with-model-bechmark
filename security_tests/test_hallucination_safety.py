import json
from pathlib import Path

from app import llm_client
from app.redaction import redact
from app.schemas import RedactedClinicalData
from benchmark.scorer import hallucination_rate


def test_missing_fields_are_not_invented_by_validation_contract(monkeypatch):
    edge_cases = json.loads((Path(__file__).parents[1] / "benchmark" / "edge_cases.json").read_text())
    missing_rates = []
    response = '{"patient_name": null, "date_of_birth": null, "diagnosis_codes": [], "medications": [], "visit_date": null, "chief_complaint": null}'
    monkeypatch.setattr(llm_client.requests, "post", lambda *args, **kwargs: type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: {"response": response}})())
    for case in edge_cases:
        redacted, _ = redact(case["raw_text"])
        result = llm_client.call_model("synthetic-test-model", redacted, RedactedClinicalData)
        assert result.is_valid
        missing_rates.append(hallucination_rate(case["missing_fields"], result.data.model_dump()))
    rate = sum(missing_rates) / len(missing_rates)
    print(f"Hallucination rate: {rate:.2%}")
    assert rate <= 0.05