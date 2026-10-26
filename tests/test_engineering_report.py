"""Regression tests for the combined engineering report."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.engineering_report import build_engineering_report, main
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


    def test_cli_prints_status_summary(self):
        findings = [
            HealthFinding("dep_ok", "pass", "info", "Matched.", "No change."),
            HealthFinding("dep_bad", "fail", "high", "Mismatch.", "Review it."),
            HealthFinding("lesson", "unknown", "medium", "Not checked.", "Verify it."),
        ]
        output = StringIO()
        with patch("tools.engineering_report.build_engineering_report", return_value=findings):
            with redirect_stdout(output):
                result = main(["--change", "test change"])
        self.assertEqual(result, 0)
        self.assertIn("PASS: 1 | WARNING: 0 | FAIL: 1 | UNKNOWN: 1", output.getvalue())

    def test_cli_hides_passing_findings_by_default(self):
        findings = [
            HealthFinding("dep_ok", "pass", "info", "Matched.", "No change."),
            HealthFinding("dep_bad", "fail", "high", "Mismatch.", "Review it."),
        ]
        output = StringIO()
        with patch("tools.engineering_report.build_engineering_report", return_value=findings):
            with redirect_stdout(output):
                main(["--change", "test change"])
        self.assertNotIn("[PASS] dep_ok", output.getvalue())
        self.assertIn("[FAIL] dep_bad", output.getvalue())

    def test_cli_all_option_displays_passing_findings(self):
        findings = [
            HealthFinding("dep_ok", "pass", "info", "Matched.", "No change."),
        ]
        output = StringIO()
        with patch("tools.engineering_report.build_engineering_report", return_value=findings):
            with redirect_stdout(output):
                main(["--change", "test change", "--all"])
        self.assertIn("[PASS] dep_ok", output.getvalue())

    def test_cli_requires_change_argument(self):
        errors = StringIO()
        with redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised:
                main([])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--change", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
