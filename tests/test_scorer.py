from benchmark.scorer import field_level_accuracy, score_models


def test_field_accuracy_handles_exact_and_fuzzy_fields():
    expected = {"patient_name": "Avery Example", "date_of_birth": "1980-01-02", "diagnosis_codes": ["Z00.00"], "medications": [], "visit_date": "2026-01-03"}
    actual = {**expected, "patient_name": "avery example"}
    assert field_level_accuracy(expected, actual) == 1.0


def test_score_models_ranks_higher_quality_model_first():
    results = [
        {"model": "good", "field_level_accuracy": 1.0, "schema_validity_first_try": True, "latency_ms": 10, "retries_needed": 0},
        {"model": "weak", "field_level_accuracy": 0.0, "schema_validity_first_try": False, "latency_ms": 20, "retries_needed": 2},
    ]
    assert score_models(results)[0]["model"] == "good"