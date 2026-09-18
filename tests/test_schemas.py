from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas import ExtractedClinicalData, schema_to_prompt_instructions


def test_extracted_data_rejects_hallucinated_fields():
    with pytest.raises(ValidationError):
        ExtractedClinicalData(
            patient_name="Fictional Patient",
            date_of_birth=date(1980, 1, 2),
            diagnosis_codes=["Z00.00"],
            medications=[],
            visit_date=date(2026, 1, 3),
            hallucinated_field="remove me",
        )


def test_schema_prompt_contains_field_descriptions():
    prompt = schema_to_prompt_instructions(ExtractedClinicalData)
    assert "patient_name" in prompt
    assert "date_of_birth" in prompt
    assert "Never infer or guess" in prompt


def test_missing_fields_default_to_null_or_empty_lists():
    data = ExtractedClinicalData()

    assert data.patient_name is None
    assert data.date_of_birth is None
    assert data.diagnosis_codes == []
    assert data.medications == []
    assert data.visit_date is None
    assert data.chief_complaint is None