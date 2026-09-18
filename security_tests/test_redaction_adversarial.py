from collections import Counter

from app.redaction import redact


CASES = [
    ("Patient Avery Example was born 1988-04-12; SSN 123-45-6789.", {"PERSON", "DATE_TIME", "US_SSN"}),
    ("The mid-sentence patient Avery Example arrived on 12-Apr-88.", {"PERSON", "DATE_TIME"}),
    ("Call (555) 123-4567 or 555.987.6543 for Avery Example.", {"PERSON", "PHONE_NUMBER"}),
    ("Avery Example saw Dr Avery Example at Clinic Road.", {"PERSON"}),
    ("The SSN words are four one five two six three seven eight nine.", {"US_SSN"}),
]


def test_adversarial_phi_is_redacted_and_prints_precision_recall_summary():
    counts = Counter()
    for note, expected_entities in CASES:
        redacted, mapping = redact(note)
        found_entities = {item["entity_type"] for item in mapping}
        assert not any(value in redacted for value in ("Avery Example", "123-45-6789", "555-123-4567"))
        for entity in expected_entities:
            counts[(entity, "tp")] += int(entity in found_entities)
            counts[(entity, "fn")] += int(entity not in found_entities)
        for entity in found_entities - expected_entities:
            counts[(entity, "fp")] += 1

    report = []
    for entity in sorted({entity for entity, _ in counts}):
        tp = counts[(entity, "tp")]
        fp = counts[(entity, "fp")]
        fn = counts[(entity, "fn")]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        report.append(f"{entity}: precision={precision:.2f}, recall={recall:.2f}")
    print("Redaction precision/recall summary: " + "; ".join(report))