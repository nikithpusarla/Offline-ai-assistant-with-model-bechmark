from collections import defaultdict
from functools import lru_cache

import spacy
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class _LocalPatternNlpEngine(NlpEngine):
    def __init__(self) -> None:
        self._nlp = spacy.blank("en")

    def load(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        document = self._nlp(text)
        token_indices = [token.idx for token in document]
        lemmas = [token.text for token in document]
        return NlpArtifacts([], document, token_indices, lemmas, self, language)

    def process_batch(self, texts, language: str, batch_size: int = 1, n_process: int = 1, **kwargs):
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        return False

    def is_punct(self, word: str, language: str) -> bool:
        return False

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return ["en"]


@lru_cache(maxsize=1)
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    analyzer = AnalyzerEngine(nlp_engine=_LocalPatternNlpEngine())
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[Pattern(name="ssn", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.95)],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="MRN",
            patterns=[Pattern(name="mrn", regex=r"\b(?:MRN|Medical\s+Record(?:\s+Number)?)\s*[:#-]?\s*[A-Z0-9]{5,12}\b", score=0.98)],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[Pattern(name="email", regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", score=0.98)],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_TIME",
            patterns=[
                Pattern(name="iso_date", regex=r"\b\d{4}-\d{2}-\d{2}\b", score=0.9),
                Pattern(name="us_date", regex=r"\b\d{1,2}/\d{1,2}/\d{4}\b", score=0.9),
                Pattern(name="named_date", regex=r"\b\d{1,2}-[A-Za-z]{3}-\d{2}\b", score=0.9),
            ],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PERSON",
            patterns=[
                Pattern(
                    name="full_name",
                    regex=r"\b[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?\b",
                    score=0.7,
                )
            ],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[
                Pattern(
                    name="spoken_ssn",
                    regex=r"\b(?:four|one|two|three|five|six|seven|eight|nine|zero)(?:\s+(?:four|one|two|three|five|six|seven|eight|nine|zero)){8}\b",
                    score=0.85,
                )
            ],
        )
    )
    return analyzer, AnonymizerEngine()


def redact(text: str) -> tuple[str, list[dict]]:
    analyzer, anonymizer = _engines()
    entities = [
        "PERSON", "DATE_TIME", "US_SSN", "PHONE_NUMBER", "MEDICAL_LICENSE",
        "LOCATION", "EMAIL_ADDRESS", "MRN",
    ]
    results = analyzer.analyze(text=text, language="en", entities=entities)
    selected = []
    occupied_until = -1
    for result in sorted(results, key=lambda item: (item.start, -(item.end - item.start))):
        if result.start < occupied_until:
            continue
        selected.append(result)
        occupied_until = result.end

    counters = defaultdict(int)
    replacements = []
    for result in selected:
        entity = result.entity_type
        counters[entity] += 1
        placeholder = f"[{entity}_{counters[entity]}]"
        anonymized = anonymizer.anonymize(
            text=text[result.start : result.end],
            analyzer_results=[
                RecognizerResult(
                    entity_type=entity,
                    start=0,
                    end=result.end - result.start,
                    score=result.score,
                )
            ],
            operators={entity: OperatorConfig("replace", {"new_value": placeholder})},
        )
        replacements.append(
            {
                "entity_type": entity,
                "placeholder": placeholder,
                "original_value": text[result.start : result.end],
                "start": result.start,
                "end": result.end,
                "replacement": anonymized.text,
            }
        )

    redacted = text
    for replacement in reversed(replacements):
        redacted = (
            redacted[: replacement["start"]]
            + replacement["replacement"]
            + redacted[replacement["end"] :]
        )
    audit_mapping = [
        {
            "entity_type": item["entity_type"],
            "placeholder": item["placeholder"],
            "original_value": item["original_value"],
        }
        for item in replacements
    ]
    return redacted, audit_mapping