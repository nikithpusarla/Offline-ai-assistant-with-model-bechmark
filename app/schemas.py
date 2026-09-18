from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClinicalNoteRequest(BaseModel):
    raw_text: str = Field(description="Synthetic raw clinical note text to extract from.")
    document_type: Literal["clinical_note"] = Field(
        description="The document type; this API accepts clinical notes only."
    )


class ExtractedClinicalData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: Optional[str] = Field(default=None, description="The patient's full name, or null if not explicitly stated.")
    date_of_birth: Optional[date] = Field(default=None, description="The patient's date of birth in YYYY-MM-DD format, or null if not explicitly stated.")
    diagnosis_codes: list[str] = Field(
        default_factory=list, description="All diagnosis codes explicitly documented in the note; empty if none are present."
    )
    medications: list[str] = Field(
        default_factory=list, description="All medications explicitly documented in the note; empty if none are present."
    )
    visit_date: Optional[date] = Field(default=None, description="The clinical visit date in YYYY-MM-DD format, or null if not explicitly stated.")
    chief_complaint: Optional[str] = Field(default=None, description="The explicitly documented chief complaint, or null if absent.")


class RedactedClinicalData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: Optional[str] = Field(default=None, description="The patient name or its redaction placeholder; null if absent.")
    date_of_birth: Optional[str] = Field(default=None, description="The date of birth or its redaction placeholder; null if absent.")
    diagnosis_codes: list[str] = Field(default_factory=list, description="All explicitly documented diagnosis codes; empty if absent.")
    medications: list[str] = Field(default_factory=list, description="All explicitly documented medications; empty if absent.")
    visit_date: Optional[str] = Field(default=None, description="The visit date or its redaction placeholder; null if absent.")
    chief_complaint: Optional[str] = Field(default=None, description="The chief complaint or its redaction placeholder; null if absent.")


class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the model output validated against the schema.")
    data: Optional[Any] = Field(
        description="Validated extracted data, or null when validation failed."
    )
    raw_output: str = Field(description="The final raw model response.")
    errors: list[str] = Field(description="Validation or transport errors collected during attempts.")
    retry_count: int = Field(description="Number of retries after the initial model attempt.")
    latency_ms: float = Field(description="Total model call latency in milliseconds.")


def schema_to_prompt_instructions(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines = [
        "Return exactly one JSON object matching this schema.",
        "Do not include markdown, commentary, or fields not listed below.",
        "If a field is not present in the text, return null or an empty list. Never infer or guess a value that is not explicitly stated.",
        "If the text contains a redaction placeholder such as [PERSON_1] or [DATE_TIME_1] for a field, return that exact placeholder rather than null.",
        "Fields:",
    ]
    for name, details in properties.items():
        required_marker = "required" if name in required else "optional"
        field_type = details.get("type", "object")
        description = details.get("description", "No description provided.")
        lines.append(f"- {name} ({field_type}, {required_marker}): {description}")
    return "\n".join(lines)