import json
from pathlib import Path

from app.schemas import ExtractedClinicalData, schema_to_prompt_instructions


def test_reproducible_prompt_fixture_matches_safety_contract():
    fixture = json.loads((Path(__file__).parents[1] / "benchmark" / "prompts.json").read_text())
    prompt = schema_to_prompt_instructions(ExtractedClinicalData)

    assert "null or an empty list" in fixture["system_contract"]
    assert "Never infer or guess" in fixture["system_contract"]
    assert "[PERSON_1]" in fixture["system_contract"]
    assert "Never infer or guess" in prompt
    assert len(fixture["synthetic_cases"]) == 2