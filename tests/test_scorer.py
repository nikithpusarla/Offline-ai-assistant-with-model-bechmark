from benchmark.scorer import (
    detect_phi_leakage,
    field_level_accuracy,
    paired_model_comparisons,
    precision_recall_f1,
    safety_gate,
    score_models,
)


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


def test_list_metrics_use_precision_recall_and_f1():
    metrics = precision_recall_f1("diagnosis_codes", ["z00.00", "I10"], ["Z00.00", "R05"])

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_safety_gate_blocks_phi_leakage_and_excess_hallucination():
    assert safety_gate(0.05, False) is True
    assert safety_gate(0.051, False) is False
    assert safety_gate(0.0, True) is False
    assert detect_phi_leakage("patient [PERSON_1]", [{"original_value": "Avery Example"}]) is False
    assert detect_phi_leakage("Avery Example", [{"original_value": "Avery Example"}]) is True


def test_score_models_reports_latency_ci_categories_and_safety():
    results = [
        {"model": "good", "example_id": "a", "field_level_accuracy": 1.0, "schema_validity_first_try": True, "latency_ms": 10, "retries_needed": 0, "hallucination_rate": 0.0, "phi_leakage_detected": False, "safety_gate_passed": True, "case_category": "complete"},
        {"model": "good", "example_id": "b", "field_level_accuracy": 0.8, "schema_validity_first_try": True, "latency_ms": 20, "retries_needed": 0, "hallucination_rate": 0.0, "phi_leakage_detected": False, "safety_gate_passed": True, "case_category": "missing_field"},
        {"model": "unsafe", "example_id": "a", "field_level_accuracy": 1.0, "schema_validity_first_try": True, "latency_ms": 5, "retries_needed": 0, "hallucination_rate": 0.0, "phi_leakage_detected": True, "safety_gate_passed": False, "case_category": "complete"},
        {"model": "unsafe", "example_id": "b", "field_level_accuracy": 1.0, "schema_validity_first_try": True, "latency_ms": 5, "retries_needed": 0, "hallucination_rate": 0.0, "phi_leakage_detected": True, "safety_gate_passed": False, "case_category": "missing_field"},
    ]

    scores = score_models(results)
    good = next(item for item in scores if item["model"] == "good")
    unsafe = next(item for item in scores if item["model"] == "unsafe")
    assert good["median_latency_ms"] == 15
    assert good["p95_latency_ms"] == 19.5
    assert "missing_field" in good["category_metrics"]
    assert good["field_accuracy_ci_95"]["low"] <= good["field_accuracy"] <= good["field_accuracy_ci_95"]["high"]
    assert unsafe["composite_score"] == 0.0


def test_paired_comparisons_use_shared_cases():
    results = [
        {"model": "a", "example_id": "x", "field_level_accuracy": 0.8},
        {"model": "b", "example_id": "x", "field_level_accuracy": 0.5},
    ]

    comparison = paired_model_comparisons(results)[0]
    assert comparison["paired_cases"] == 1
    assert comparison["mean_field_accuracy_difference"] == 0.30000000000000004