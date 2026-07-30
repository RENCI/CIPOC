import unittest

from pydantic import ValidationError

from cipoc.agents.note_scanner import (
    ConceptFinding,
    ConceptFindingList,
    NoteScannerAgent,
    ScannerState,
)
from cipoc.models import (
    ClinicalNote,
    ConfidenceLevel,
    CONCEPT_DESCRIPTIONS,
    TextSpan,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def structured(self, schema, messages, **kwargs):
        self.messages = messages
        return self.response


def _finding(name, present=False):
    return ConceptFinding(
        concept=name,
        presence=present,
        confidence="max",
        evidence=[TextSpan(id=1, text="carcinoma")] if present else [],
    )


def _scanner_with(response):
    scanner = object.__new__(NoteScannerAgent)
    scanner.agent = FakeLLM(response)
    return scanner


def _state():
    return ScannerState(
        note=ClinicalNote(
            note_id=1,
            date="2025-01-01",
            type="Pathology Report",
            content="Final diagnosis: invasive carcinoma.",
        ),
        messages=[],
    )


class ConceptDetectionTests(unittest.TestCase):
    def test_concept_template_contains_every_description(self):
        self.assertEqual(
            NoteScannerAgent._concepts_to_findings(),
            [
                {"concept": name, "description": description}
                for name, description in CONCEPT_DESCRIPTIONS.items()
            ],
        )

    def test_finding_requires_model_populated_fields(self):
        with self.assertRaises(ValidationError):
            ConceptFinding(concept="cancer")

    def test_detect_concepts_normalizes_an_incomplete_response(self):
        scanner = _scanner_with(
            ConceptFindingList(findings=[_finding("surgery", present=True)])
        )

        with self.assertLogs("cipoc.agents.note_scanner", level="WARNING"):
            result = scanner.detect_concepts(_state())

        concepts = result["concepts"]
        self.assertEqual(set(concepts), set(CONCEPT_DESCRIPTIONS))
        self.assertTrue(concepts["surgery"].presence)
        self.assertTrue(concepts["cancer"].presence)
        self.assertEqual(concepts["cancer"].confidence, ConfidenceLevel.LOW)

    def test_detect_concepts_prefers_positive_duplicate(self):
        response = ConceptFindingList(
            findings=[_finding(name) for name in CONCEPT_DESCRIPTIONS]
            + [_finding("cancer", present=True)]
        )
        scanner = _scanner_with(response)

        with self.assertLogs("cipoc.agents.note_scanner", level="WARNING"):
            result = scanner.detect_concepts(_state())

        self.assertTrue(result["concepts"]["cancer"].presence)

    def test_detect_concepts_passes_template_and_returns_findings(self):
        response = ConceptFindingList(
            findings=[
                _finding(name, present=name == "cancer")
                for name in CONCEPT_DESCRIPTIONS
            ]
        )
        scanner = _scanner_with(response)

        result = scanner.detect_concepts(_state())

        self.assertTrue(result["concepts"]["cancer"].presence)
        self.assertEqual(set(result["concepts"]), set(CONCEPT_DESCRIPTIONS))
        prompt = scanner.agent.messages[-1].content
        for name, description in CONCEPT_DESCRIPTIONS.items():
            self.assertIn(f'"concept": "{name}"', prompt)
            self.assertIn(description, prompt)


if __name__ == "__main__":
    unittest.main()
