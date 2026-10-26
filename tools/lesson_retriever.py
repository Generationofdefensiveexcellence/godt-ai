"""Explainable retrieval of engineering lessons from Markdown."""

from dataclasses import dataclass
from pathlib import Path
import re


DEFAULT_LESSONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "engineering"
    / "LESSONS_LEARNED.md"
)


@dataclass(frozen=True)
class RetrievedLesson:
    lesson_id: str
    title: str
    text: str
    matched_terms: tuple[str, ...]
    score: float


def _parse_lessons(markdown: str) -> list[tuple[str, str, str]]:
    """Extract lesson ID, title, and text from L-prefixed Markdown headings."""
    headings = list(
        re.finditer(
            r"(?m)^#{2,4}\s+(L\d+)\s*[—–-]\s*(.+?)\s*$",
            markdown,
        )
    )
    lessons = []

    for index, heading in enumerate(headings):
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        body = markdown[heading.end():end].strip()
        lessons.append(
            (heading.group(1), heading.group(2).strip(), body)
        )

    return lessons


def retrieve_lessons(
    query: str,
    lessons_path: str | Path = DEFAULT_LESSONS_PATH,
    *,
    min_score: float = 0.15,
    min_matched_terms: int = 2,
    limit: int = 3,
) -> list[RetrievedLesson]:
    """Retrieve lessons using explainable token overlap, not semantic understanding."""
    if not isinstance(query, str) or not query.strip():
        return []
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1")
    if min_matched_terms < 1:
        raise ValueError("min_matched_terms must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    path = Path(lessons_path)
    if not path.is_file():
        return []

    lessons = _parse_lessons(path.read_text(encoding="utf-8"))
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
        "our", "that", "the", "this", "to", "was", "we", "what", "when",
        "with", "not", "no", "but", "all", "about", "reported",
    }

    def tokens(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in stop_words
        }

    query_terms = tokens(query)
    if not query_terms:
        return []

    results = []
    for lesson_id, title, body in lessons:
        title_terms = tokens(title)

        # Exclude metadata from relevance scoring, but preserve it in the
        # returned lesson text for traceability.
        content_lines = []
        for line in body.splitlines():
            if re.match(
                r"^\*\*(?:Source|Applies to|Verification status):\*\*",
                line.strip(),
                flags=re.IGNORECASE,
            ):
                continue
            content_lines.append(line)

        body_terms = tokens("\n".join(content_lines))
        title_matches = query_terms & title_terms
        body_matches = query_terms & body_terms
        matched = tuple(sorted(title_matches | body_matches))

        # Title matches carry more weight than incidental body-word overlap.
        score = (
            2 * len(title_matches) + len(body_matches)
        ) / (2 * len(query_terms))

        if len(matched) >= min_matched_terms and score >= min_score:
            results.append(
                RetrievedLesson(
                    lesson_id=lesson_id,
                    title=title,
                    text=body,
                    matched_terms=matched,
                    score=min(score, 1.0),
                )
            )

    results.sort(key=lambda item: (-item.score, item.lesson_id))
    return results[:limit]
