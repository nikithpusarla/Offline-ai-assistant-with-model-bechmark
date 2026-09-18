import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm_client import call_model
from app.redaction import redact
from app.schemas import RedactedClinicalData
from benchmark.scorer import field_level_accuracy, hallucination_rate, score_models


def _restore(data: dict, mapping: list[dict]) -> dict:
    restored = {}
    for field, value in data.items():
        if isinstance(value, str):
            for item in mapping:
                value = value.replace(item["placeholder"], item["original_value"])
        elif isinstance(value, list):
            value = [
                next((item["original_value"] if item["placeholder"] in item_value else item_value for item in mapping), item_value)
                for item_value in value
            ]
        restored[field] = value
    return restored


def run() -> list[dict]:
    config = yaml.safe_load((ROOT / "models_config.yaml").read_text())
    golden_set = json.loads((ROOT / "benchmark" / "golden_set.json").read_text())
    results_path = ROOT / "benchmark" / "results.json"
    raw_results = json.loads(results_path.read_text()) if results_path.exists() else []
    completed = {(item["model"], item["example_id"]) for item in raw_results}
    for model_config in config["models"]:
        model_name = model_config["name"]
        for example in golden_set:
            if (model_name, example["id"]) in completed:
                continue
            print(f"Running {model_name} / {example['id']}...", flush=True)
            redacted_text, mapping = redact(example["raw_text"])
            result = call_model(model_name, redacted_text, RedactedClinicalData)
            actual = _restore(result.data.model_dump() if result.data else {}, mapping)
            missing_fields = [field for field, value in example["expected"].items() if value in (None, [])]
            raw_results.append({
                "model": model_name,
                "example_id": example["id"],
                "schema_validity_first_try": result.retry_count == 0 and result.is_valid,
                "retries_needed": result.retry_count,
                "latency_ms": result.latency_ms,
                "field_level_accuracy": field_level_accuracy(example["expected"], actual),
                "hallucination_rate": hallucination_rate(missing_fields, actual),
                "redacted_entities": len(mapping),
            })
            results_path.write_text(json.dumps(raw_results, indent=2))
            results_path.write_text(json.dumps(raw_results, indent=2))
    return raw_results


def print_scores(raw_results: list[dict]) -> None:
    scores = score_models(raw_results)
    table = Table(title="Local Model Benchmark")
    columns = ["model", "field_accuracy", "first_try_rate", "latency_score", "average_retries", "hallucination_rate", "composite_score"]
    for column in columns:
        table.add_column(column)
    for score in scores:
        table.add_row(*(score["model"] if key == "model" else f"{score[key]:.3f}" for key in columns))
    Console().print(table)
    markdown = [
        "| Model | Field accuracy | First-try validity | Latency score | Avg retries | Hallucination rate | Composite score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    markdown.extend(
        f"| {score['model']} | {score['field_accuracy']:.3f} | {score['first_try_rate']:.3f} | {score['latency_score']:.3f} | {score['average_retries']:.3f} | {score['hallucination_rate']:.3f} | {score['composite_score']:.3f} |"
        for score in scores
    )
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text()
    start_marker = "<!-- BENCHMARK_TABLE_START -->"
    end_marker = "<!-- BENCHMARK_TABLE_END -->"
    if start_marker in readme and end_marker in readme:
        prefix = readme.split(start_marker, 1)[0]
        suffix = readme.split(end_marker, 1)[1]
        readme_path.write_text(prefix + start_marker + "\n" + "\n".join(markdown) + "\n" + end_marker + suffix)


if __name__ == "__main__":
    print_scores(run())