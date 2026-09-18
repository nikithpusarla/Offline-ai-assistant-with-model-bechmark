from difflib import SequenceMatcher


def _similar(left, right) -> float:
    if isinstance(left, list) and isinstance(right, list):
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return sum(max(SequenceMatcher(None, str(item).lower(), str(candidate).lower()).ratio() for candidate in right) for item in left) / len(left)
    return SequenceMatcher(None, str(left).lower(), str(right).lower()).ratio()


def field_level_accuracy(expected: dict, actual: dict) -> float:
    fuzzy_fields = {"patient_name", "medications"}
    scores = []
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if field in fuzzy_fields:
            scores.append(_similar(expected_value, actual_value))
        else:
            scores.append(1.0 if actual_value == expected_value else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def hallucination_rate(expected_missing_fields: list[str], actual: dict) -> float:
    if not expected_missing_fields:
        return 0.0
    invented = sum(
        1 for field in expected_missing_fields
        if actual.get(field) not in (None, [], "")
    )
    return invented / len(expected_missing_fields)


def score_models(results: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["model"], []).append(result)
    latencies = [item["latency_ms"] for item in results] or [1.0]
    max_latency = max(latencies)
    min_latency = min(latencies)
    max_retries = max((item["retries_needed"] for item in results), default=1)
    scored = []
    for model, items in grouped.items():
        accuracy = sum(item["field_level_accuracy"] for item in items) / len(items)
        first_try = sum(item["schema_validity_first_try"] for item in items) / len(items)
        average_latency = sum(item["latency_ms"] for item in items) / len(items)
        latency_score = 1.0 if max_latency == min_latency else 1 - ((average_latency - min_latency) / (max_latency - min_latency))
        average_retries = sum(item["retries_needed"] for item in items) / len(items)
        retry_score = 1.0 if max_retries == 0 else 1 - (average_retries / max_retries)
        hallucination = sum(item.get("hallucination_rate", 0.0) for item in items) / len(items)
        hallucination_score = 1 - hallucination
        composite = (0.35 * accuracy + 0.20 * first_try + 0.20 * latency_score
                 + 0.15 * retry_score + 0.10 * hallucination_score)
        scored.append({"model": model, "field_accuracy": accuracy, "first_try_rate": first_try, "latency_score": latency_score, "average_retries": average_retries, "hallucination_rate": hallucination, "composite_score": composite})
    return sorted(scored, key=lambda item: item["composite_score"], reverse=True)