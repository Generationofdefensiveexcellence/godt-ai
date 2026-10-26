
"""Read-only engineering preflight based on explicit lesson rules."""

from tools.health_findings import HealthFinding
from tools.lesson_retriever import retrieve_lessons


LESSON_CHECKS = {
    "L001": (
        (
            "provider_failure_truthfulness",
            "high",
            "Provider failures must be reported truthfully.",
            "Add or inspect tests proving failed AI generation is not "
            "reported as successful.",
        ),
        (
            "provider_failure_user_message",
            "medium",
            "Users must not be shown fallback text as a genuine AI reply.",
            "Verify the user-facing failure behavior with a regression test.",
        ),
        (
            "provider_failure_safe_logging",
            "medium",
            "Provider failures must produce safe diagnostic details.",
            "Inspect logging and verify that secrets and sensitive data "
            "are not exposed.",
        ),
        (
            "provider_failure_regression_tests",
            "high",
            "Provider failure paths must have regression coverage.",
            "Test primary-provider success, fallback-provider success, "
            "and all-provider failure.",
        ),
    ),
    "L002": (
        (
            "lesson_application_evidence",
            "high",
            "A retrieved lesson must be applied to the proposed change.",
            "Identify the exact code change that addresses the lesson.",
        ),
        (
            "lesson_verification_evidence",
            "high",
            "The proposed change must be verified with evidence or tests.",
            "Run relevant regression tests and record their results.",
        ),
        (
            "lesson_correction_record",
            "low",
            "New findings and corrections should be recorded.",
            "Update the engineering journal after evidence is collected.",
        ),
    ),
}


def engineering_preflight(
    proposed_change: str,
    *,
    lessons_path=None,
) -> list[HealthFinding]:
    """Return unverified checks for lessons relevant to a proposed change."""
    if not isinstance(proposed_change, str) or not proposed_change.strip():
        raise ValueError("proposed_change must be a non-empty string")

    retrieval_options = {}
    if lessons_path is not None:
        retrieval_options["lessons_path"] = lessons_path

    lessons = retrieve_lessons(proposed_change, **retrieval_options)

    if not lessons:
        return [
            HealthFinding(
                check="lesson_retrieval",
                status="unknown",
                severity="info",
                evidence="No relevant lesson was retrieved for the proposed change.",
                recommendation=(
                    "Review the proposed change and lessons manually; "
                    "no conclusion about applicability can be drawn."
                ),
                verified=False,
            )
        ]

    findings = []
    for lesson in lessons:
        checks = LESSON_CHECKS.get(lesson.lesson_id)
        if checks is None:
            findings.append(
                HealthFinding(
                    check=f"{lesson.lesson_id}_manual_review",
                    status="unknown",
                    severity="info",
                    evidence=(
                        f"Retrieved {lesson.lesson_id}: {lesson.title}. "
                        f"Matched terms: {', '.join(lesson.matched_terms)}."
                    ),
                    recommendation=(
                        "Review this lesson and define explicit checks "
                        "before relying on it."
                    ),
                    verified=False,
                )
            )
            continue

        for check, severity, requirement, recommendation in checks:
            findings.append(
                HealthFinding(
                    check=check,
                    status="unknown",
                    severity=severity,
                    evidence=(
                        f"Requirement derived from retrieved "
                        f"{lesson.lesson_id}: {requirement} "
                        f"Matched terms: {', '.join(lesson.matched_terms)}. "
                        "No implementation or test evidence was evaluated."
                    ),
                    recommendation=recommendation,
                    verified=False,
                )
            )

    return findings
