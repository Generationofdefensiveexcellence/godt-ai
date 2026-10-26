"""Regression tests for the combined engineering report."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.engineering_report import build_engineering_report
from tools.health_findings import HealthFinding


class EngineeringReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        self.requirements_path = self.root / "requirements.txt"
        self.requirements_path.write_text(
            "example-package==1.2.3\n",
            encoding="utf-8",
        )

        self.lessons_path = self.root / "LESSONS_LEARNED.md"
        self.lessons_path.write_text(
            "# Engineering lessons\n\n"
            "## L001 — Do not confuse fallback content with successful AI generation\n"
            "When all AI providers fail, report failure truthfully. "
            "Do not report fallback text as a successful generated reply. "
            "Show a clear failure message, log safe diagnostics, "
            "and test provider failure paths.\n",
            encoding="utf-8",
        )

    def test_combines_dependency_and_lesson_findings(self):
        dependency_finding = HealthFinding(
            check="dependency_presence",
            status="fail",
            severity="high",
            evidence="Example package is not installed.",
            recommendation="Review the active environment.",
        )

        with patch(
            "tools.engineering_report.compare_requirements_file",
            return_value=[dependency_finding],
        ):
            findings = build_engineering_report(
                "AI provider failure and fallback reply reported as success",
                requirements_path=self.requirements_path,
                lessons_path=self.lessons_path,
            )

        checks = {finding.check for finding in findings}
        self.assertIn("dependency_presence", checks)
        self.assertIn("provider_failure_truthfulness", checks)

    def test_dependency_failure_is_preserved(self):
        dependency_finding = HealthFinding(
            check="dependency_presence",
            status="fail",
            severity="high",
            evidence="A required package is missing.",
            recommendation="Review the environment.",
        )

        with patch(
            "tools.engineering_report.compare_requirements_file",
            return_value=[dependency_finding],
        ):
            findings = build_engineering_report(
                "unrelated proposal",
                requirements_path=self.requirements_path,
                lessons_path=self.lessons_path,
            )

        result = next(
            finding for finding in findings
            if finding.check == "dependency_presence"
        )
        self.assertEqual(result.status, "fail")
        self.assertFalse(result.verified)

    def test_lesson_findings_remain_unverified(self):
        with patch(
            "tools.engineering_report.compare_requirements_file",
            return_value=[],
        ):
            findings = build_engineering_report(
                "AI provider failure and fallback reply reported as success",
                requirements_path=self.requirements_path,
                lessons_path=self.lessons_path,
            )

        self.assertTrue(findings)
        self.assertTrue(all(
            not finding.verified
            for finding in findings
            if finding.check.startswith("provider_failure_")
        ))
        self.assertTrue(all(
            finding.status == "unknown"
            for finding in findings
            if finding.check.startswith("provider_failure_")
        ))


if __name__ == "__main__":
    unittest.main()
