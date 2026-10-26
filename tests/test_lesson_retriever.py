"""Tests for explainable engineering lesson retrieval."""

from pathlib import Path
import tempfile
import unittest

from tools.lesson_retriever import retrieve_lessons


class LessonRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.lessons_path = Path(self.temp_dir.name) / "LESSONS_LEARNED.md"
        self.lessons_path.write_text(
            "# Engineering lessons\n\n"
            "## L001 — Do not confuse fallback content with successful AI generation\n"
            "When all AI providers fail, report failure truthfully. "
            "Do not report fallback text as a successful generated reply. "
            "Test provider failure and response status.\n\n"
            "## L002 — A journal entry is not proof of learning\n"
            "Retrieve relevant lessons, apply them to the proposed action, "
            "and verify the result with evidence and tests.\n",
            encoding="utf-8",
        )

    def test_retrieves_provider_fallback_lesson(self):
        results = retrieve_lessons(
            "AI provider fails but fallback reply is reported as success",
            self.lessons_path,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].lesson_id, "L001")
        self.assertTrue(results[0].matched_terms)
        self.assertGreater(results[0].score, 0)

    def test_retrieves_lesson_about_proving_learning(self):
        results = retrieve_lessons(
            "retrieve lessons and verify learning with evidence and tests",
            self.lessons_path,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].lesson_id, "L002")

    def test_journal_query_ranks_lesson_two_first(self):
        results = retrieve_lessons(
            "a journal entry does not prove learning",
            self.lessons_path,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].lesson_id, "L002")
        self.assertNotIn("not", results[0].matched_terms)

    def test_journal_query_excludes_weak_l001_match(self):
        results = retrieve_lessons(
            "a journal entry does not prove learning",
            self.lessons_path,
        )
        self.assertEqual([result.lesson_id for result in results], ["L002"])

    def test_real_lessons_file_excludes_metadata_false_positive(self):
        real_lessons = Path(__file__).resolve().parents[1] / "docs" / "engineering" / "LESSONS_LEARNED.md"
        if not real_lessons.is_file():
            self.skipTest("Real engineering lessons file is unavailable")

        results = retrieve_lessons(
            "a journal entry does not prove learning",
            real_lessons,
        )
        self.assertEqual([result.lesson_id for result in results], ["L002"])

    def test_min_matched_terms_must_be_positive(self):
        with self.assertRaises(ValueError):
            retrieve_lessons(
                "AI provider failure",
                self.lessons_path,
                min_matched_terms=0,
            )

    def test_unrelated_query_returns_no_lessons(self):
        results = retrieve_lessons(
            "underwater photography coral reefs",
            self.lessons_path,
        )
        self.assertEqual(results, [])

    def test_empty_query_returns_no_lessons(self):
        self.assertEqual(retrieve_lessons("   ", self.lessons_path), [])

    def test_missing_file_returns_no_lessons(self):
        missing = Path(self.temp_dir.name) / "missing.md"
        self.assertEqual(
            retrieve_lessons("AI provider failure", missing),
            [],
        )

    def test_invalid_min_score_is_rejected(self):
        with self.assertRaises(ValueError):
            retrieve_lessons("AI provider failure", self.lessons_path,
                             min_score=1.1)

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            retrieve_lessons("AI provider failure", self.lessons_path,
                             limit=0)


if __name__ == "__main__":
    unittest.main()
