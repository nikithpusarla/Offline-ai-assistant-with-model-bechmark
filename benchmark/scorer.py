from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from datetime import date
from statistics import mean, median
from typing import Any


SAFETY_HALLUCINATION_THRESHOLD = 0.05
BOOTSTRAP_SAMPLES = 2000


def normalize_field_value(field: str, value: Any) -> str:
    """Normalize one scalar according to the semantics of its extraction field."""
    if value is None:
        return ""
    text = str(value).strip().casefold()
    if field in {"date_of_birth", "visit_date"}:
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return text
    if field == "diagnosis_codes":
        return re.sub(r"\s+", "", text).upper()
    if field == "medications":
        text = re.sub(r"[^\w.]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    if field in {"patient_name", "chief_complaint"}:
        return re.sub(r"\s+", " ", text)
    return re.sub(r"\s+", " ", text)


def _as_values(field: str, value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return sorted({normalized for normalized in (normalize_field_value(field, item) for item in values) if normalized})


def precision_recall_f1(field: str, expected: Any, actual: Any) -> dict[str, float]:
    expected_values = set(_as_values(field, expected))
    actual_values = set(_as_values(field, actual))
    true_positives = len(expected_values & actual_values)
    precision = true_positives / len(actual_values) if actual_values else (1.0 if not expected_values else 0.0)
    recall = true_positives / len(expected_values) if expected_values else (1.0 if not actual_values else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def field_metrics(expected: dict, actual: dict) -> dict[str, dict[str, float]]:
    return {
        field: precision_recall_f1(field, expected.get(field), actual.get(field))
        for field in expected
    }


def field_level_accuracy(expected: dict, actual: dict) -> float:
    metrics = field_metrics(expected, actual)
    return mean(item["f1"] for item in metrics.values()) if metrics else 0.0


def hallucination_rate(expected_missing_fields: list[str], actual: dict) -> float:
    if not expected_missing_fields:
        return 0.0
    invented = sum(1 for field in expected_missing_fields if actual.get(field) not in (None, [], ""))
    return invented / len(expected_missing_fields)


def detect_phi_leakage(raw_output: str, redaction_mapping: list[dict]) -> bool:
    """Return true when an original redacted value appears in the model response."""
    output = raw_output.casefold()
    return any(item.get("original_value", "").casefold() in output for item in redaction_mapping if item.get("original_value"))


def safety_gate(hallucination: float, phi_leakage: bool) -> bool:
    return not phi_leakage and hallucination <= SAFETY_HALLUCINATION_THRESHOLD


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_ci(values: list[float], seed: int = 7) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    generator = random.Random(seed)
    samples = [mean(generator.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES)]
    return {"low": _percentile(samples, 0.025), "high": _percentile(samples, 0.975)}


def _category_metrics(items: list[dict]) -> dict[str, dict[str, float]]:
    categories: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        categories[item.get("case_category", "uncategorized")].append(item)
    return {
        category: {
            "cases": len(category_items),
            "field_accuracy": mean(item["field_level_accuracy"] for item in category_items),
            "hallucination_rate": mean(item.get("hallucination_rate", 0.0) for item in category_items),
        }
        for category, category_items in sorted(categories.items())
    }


def _field_metrics(items: list[dict]) -> dict[str, dict[str, float]]:
    fields: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in items:
        for field, metrics in item.get("field_metrics", {}).items():
            fields[field].append(metrics)
    return {
        field: {
            metric: mean(values[metric] for values in metric_items)
            for metric in ("precision", "recall", "f1")
        }
        for field, metric_items in sorted(fields.items())
    }


def paired_model_comparisons(results: list[dict]) -> list[dict]:
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in results:
        by_model[item["model"]][item["example_id"]] = item
    models = sorted(by_model)
    comparisons = []
    for index, left_model in enumerate(models):
        for right_model in models[index + 1 :]:
            common_ids = sorted(set(by_model[left_model]) & set(by_model[right_model]))
            differences = [
                by_model[left_model][case_id]["field_level_accuracy"]
                - by_model[right_model][case_id]["field_level_accuracy"]
                for case_id in common_ids
            ]
            comparisons.append({
                "left_model": left_model,
                "right_model": right_model,
                "paired_cases": len(common_ids),
                "mean_field_accuracy_difference": mean(differences) if differences else 0.0,
                "difference_ci_95": _bootstrap_ci(differences),
            })
    return comparisons


def score_models(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        grouped[result["model"]].append(result)
    all_latencies = [item["latency_ms"] for item in results]
    p95_by_model = {model: _percentile([item["latency_ms"] for item in items], 0.95) for model, items in grouped.items()}
    min_p95 = min(p95_by_model.values(), default=0.0)
    max_p95 = max(p95_by_model.values(), default=0.0)
    max_retries = max((item["retries_needed"] for item in results), default=0)
    scored = []
    for model, items in grouped.items():
        accuracy_values = [item["field_level_accuracy"] for item in items]
        accuracy = mean(accuracy_values)
        first_try = mean(float(item["schema_validity_first_try"]) for item in items)
        latencies = [item["latency_ms"] for item in items]
        p95_latency = p95_by_model[model]
        latency_score = 1.0 if max_p95 == min_p95 else 1 - ((p95_latency - min_p95) / (max_p95 - min_p95))
        average_retries = mean(item["retries_needed"] for item in items)
        retry_score = 1.0 if max_retries == 0 else 1 - (average_retries / max_retries)
        hallucination = mean(item.get("hallucination_rate", 0.0) for item in items)
        phi_leakage_rate = mean(float(item.get("phi_leakage_detected", False)) for item in items)
        gate_passed = all(item.get("safety_gate_passed", True) for item in items)
        safety_score = 1.0 - hallucination
        composite = (0.35 * accuracy + 0.20 * first_try + 0.20 * latency_score + 0.15 * retry_score + 0.10 * safety_score)
        if not gate_passed:
            composite = 0.0
        scored.append({
            "model": model,
            "field_accuracy": accuracy,
            "field_accuracy_ci_95": _bootstrap_ci(accuracy_values),
            "first_try_rate": first_try,
            "median_latency_ms": median(latencies),
            "p95_latency_ms": p95_latency,
            "latency_score": latency_score,
            "average_retries": average_retries,
            "hallucination_rate": hallucination,
            "phi_leakage_rate": phi_leakage_rate,
            "safety_gate_passed": gate_passed,
            "field_metrics": _field_metrics(items),
            "category_metrics": _category_metrics(items),
            "composite_score": composite,
        })
    return sorted(scored, key=lambda item: item["composite_score"], reverse=True)
