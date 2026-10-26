"""Combine dependency and lesson findings into a read-only engineering report."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main(argv=None) -> int:
    """Print a read-only engineering report for a proposed change."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a read-only GODT AI engineering report."
    )
    parser.add_argument(
        "--change",
        required=True,
        help="Describe the proposed engineering change.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Display passing findings as well as actionable findings.",
    )
    args = parser.parse_args(argv)

    try:
        findings = build_engineering_report(args.change)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("GODT AI ENGINEERING REPORT")
    print(f"Proposed change: {args.change}")
    print(f"Total findings: {len(findings)}")

    counts = {
        status: sum(finding.status.lower() == status for finding in findings)
        for status in ("pass", "warning", "fail", "unknown")
    }
    print(
        "Summary: "
        + " | ".join(
            f"{status.upper()}: {count}" for status, count in counts.items()
        )
    )

    displayed_findings = findings if args.all else [
        finding for finding in findings if finding.status.lower() != "pass"
    ]
    if not args.all:
        print("Showing non-passing findings only (use --all to show everything).")
    if not displayed_findings:
        print("\nNo actionable findings to display.")

    for finding in displayed_findings:
        print(
            f"\n[{finding.status.upper()}] "
            f"{finding.check} | severity={finding.severity} "
            f"| verified={finding.verified}"
        )
        print(f"Evidence: {finding.evidence}")
        print(f"Recommendation: {finding.recommendation}")

    print("\nReport complete. Findings are not automatically verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
