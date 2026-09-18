from app.redaction import redact


def test_redact_replaces_synthetic_phi_with_numbered_placeholders():
    note = "Patient: Avery Example, DOB: 1988-04-12. SSN: 123-45-6789. MRN: MRN-ABC12345. Email: fake@example.test."

    redacted, mapping = redact(note)

    assert "Avery Example" not in redacted
    assert "1988-04-12" not in redacted
    assert "123-45-6789" not in redacted
    assert "ABC12345" not in redacted
    assert "fake@example.test" not in redacted
    assert "[PERSON_1]" in redacted
    assert "[DATE_TIME_1]" in redacted
    assert "[US_SSN_1]" in redacted
    assert {item["entity_type"] for item in mapping} >= {"PERSON", "DATE_TIME", "US_SSN", "MRN", "EMAIL_ADDRESS"}