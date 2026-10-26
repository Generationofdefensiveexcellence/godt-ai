"""Combine dependency and lesson findings into a read-only engineering report."""

from pathlib import Path

from tools.dependency_diagnostics import compare_requirements_file
from tools.engineering_preflight import engineering_preflight
from tools.health_findings import HealthFinding


def build_engineering_report(
    proposed_change: str,
    *,
    requirements_path: Path | None = None,
    lessons_path: Path | None = None,
) -> list[HealthFinding]:
    """Combine dependency diagnostics and lesson checks without modifying the project."""
    findings: list[HealthFinding] = []

    if requirements_path is not None:
        dependency_findings = compare_requirements_file(requirements_path)
    else:
        dependency_findings = compare_requirements_file(
            Path(__file__).resolve().parent.parent / "requirements.txt"
        )

    findings.extend(dependency_findings)

    lesson_options = {}
    if lessons_path is not None:
        lesson_options["lessons_path"] = lessons_path

    findings.extend(
        engineering_preflight(proposed_change, **lesson_options)
    )

    return findings
