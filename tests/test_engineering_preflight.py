
"""Regression tests for the read-only engineering preflight."""

from pathlib import Path
import tempfile
import unittest

from tools.engineering_preflight import engineering_preflight


class EngineeringPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.lessons_path = Path(self.temp_dir.name) / "LESSONS_LEARNED.md"
        self.lessons_path.write_text(
            "# Engineering lessons\n\n"
            "## L001 — Do not confuse fallback content with successful AI generation\n"
            "When all AI providers fail, report failure truthfully. "
            "Do not report fallback text as a successful generated reply. "
            "Show a clear failure message, log safe diagnostics, "
            "and test provider failure paths.\n\n"
            "## L002 — A journal entry is not proof of learning\n"
            "Retrieve relevant lessons, apply them to the proposed action, "
            "and verify the result with evidence and tests. "
            "Record new findings and corrections.\n",
            encoding="utf-8",
        )

    def test_provider_failure_generates_explicit_checks(self):
        findings = engineering_preflight(
            "AI provider failure and fallback reply reported as success",
            lessons_path=self.lessons_path,
        )
        checks = {finding.check for finding in findings}

        self.assertIn("provider_failure_truthfulness", checks)
        self.assertIn("provider_failure_user_message", checks)
        self.assertIn("provider_failure_safe_logging", checks)
        self.assertIn("provider_failure_regression_tests", checks)

    def test_learning_query_generates_evidence_checks(self):
        findings = engineering_preflight(
            "retrieve lessons and verify learning with evidence and tests",
            lessons_path=self.lessons_path,
        )
        checks = {finding.check for finding in findings}

        self.assertIn("lesson_application_evidence", checks)
        self.assertIn("lesson_verification_evidence", checks)

    def test_retrieval_never_marks_findings_verified(self):
        findings = engineering_preflight(
            "AI provider failure and fallback reply reported as success",
            lessons_path=self.lessons_path,
        )

        self.assertTrue(findings)
        self.assertTrue(all(finding.verified is False for finding in findings))
        self.assertTrue(all(finding.status == "unknown" for finding in findings))

    def test_empty_proposal_is_rejected(self):
        with self.assertRaises(ValueError):
            engineering_preflight(
                "   ",
                lessons_path=self.lessons_path,
            )

    def test_unrelated_proposal_does_not_create_lesson_requirements(self):
        findings = engineering_preflight(
            "underwater photography coral reefs",
            lessons_path=self.lessons_path,
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].check, "lesson_retrieval")
        self.assertEqual(findings[0].status, "unknown")
        self.assertFalse(findings[0].verified)


if __name__ == "__main__":
    unittest.main()
