"""Unit tests for Kickstarter SQLite scrape consolidation. Created: 2025-06-15"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from processing.sqlite_schema import (  # noqa: E402
    compute_item_status,
    connect_db,
    extract_slug,
    get_projects_needing_scrape,
    init_schema,
    insert_comment,
    insert_update,
    parse_batch_timestamp,
    refresh_completeness_status,
    refresh_scraped_counts,
    replace_comments_for_project,
    update_project_ks_counts,
    upsert_project,
)
from scrapers.nav_counts import parse_nav_counts_from_html  # noqa: E402
from scrapers.ks_session import (  # noqa: E402
    is_cloudflare_challenge,
    parse_csrf_from_html,
)


SAMPLE_HTML = """
<html><body>
<a id="comments-emoji" class="tabbed-nav__link"
   data-comments-count="184"
   emoji-data="&lt;data class=&quot;Project1065095549&quot; data-value=&quot;181&quot; data-format=&quot;number&quot; itemprop=&quot;Project[comments_count]&quot;&gt;181&lt;/data&gt;">
   Comments 181
</a>
<a id="updates-emoji" class="tabbed-nav__link" emoji-data="0" href="/projects/foo/posts">
   Updates 0
</a>
</body></html>
"""


class TestKsSession(unittest.TestCase):
    def test_is_cloudflare_challenge(self):
        self.assertTrue(is_cloudflare_challenge("<html><title>Just a moment...</title></html>"))
        self.assertFalse(
            is_cloudflare_challenge(
                '<html><meta name="csrf-token" content="abc123"></html>'
            )
        )

    def test_parse_csrf_from_meta(self):
        html = '<html><head><meta name="csrf-token" content="token-xyz"></head></html>'
        self.assertEqual(parse_csrf_from_html(html), "token-xyz")

    def test_parse_csrf_from_json_fallback(self):
        html = '<script>window.bootstrap = {"csrfToken":"json-token"}</script>'
        self.assertEqual(parse_csrf_from_html(html), "json-token")


class TestNavCounts(unittest.TestCase):
    def test_parse_nav_counts_from_html(self):
        counts = parse_nav_counts_from_html(SAMPLE_HTML)
        self.assertEqual(counts.ks_comments_nav, 184)
        self.assertEqual(counts.ks_comments_emoji, 181)
        self.assertEqual(counts.ks_updates_nav, 0)

    def test_parse_updates_from_link_text(self):
        html = '<a id="updates-emoji">Updates 12</a>'
        counts = parse_nav_counts_from_html(html)
        self.assertEqual(counts.ks_updates_nav, 12)


class TestSqliteSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"
        self.conn = connect_db(self.db_path)
        init_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_extract_slug(self):
        url = "https://www.kickstarter.com/projects/liiton/proclad-a-hybrid-nonstick-pan"
        self.assertEqual(extract_slug(url), "liiton/proclad-a-hybrid-nonstick-pan")

    def test_parse_batch_timestamp(self):
        ts = parse_batch_timestamp("kickstarter_comments_batch_3_4829101_20250615_143022.csv")
        self.assertIn("2025-06-15", ts)

    def test_compute_item_status(self):
        self.assertEqual(compute_item_status(0, 0), "expected_zero")
        self.assertEqual(compute_item_status(10, 0), "missing")
        self.assertEqual(compute_item_status(10, 10), "complete")
        self.assertEqual(compute_item_status(10, 7), "partial")
        self.assertEqual(compute_item_status(None, 5), "not_checked")

    def test_insert_and_refresh_counts(self):
        upsert_project(
            self.conn,
            "p1",
            "https://www.kickstarter.com/projects/creator/proj",
        )
        insert_comment(
            self.conn,
            {
                "id": "c1",
                "project_id": "p1",
                "parent_id": None,
                "body": "hello",
            },
        )
        insert_comment(
            self.conn,
            {
                "id": "c2",
                "project_id": "p1",
                "parent_id": "c1",
                "body": "reply",
            },
        )
        self.conn.commit()
        refresh_scraped_counts(self.conn, project_id="p1")
        row = self.conn.execute(
            "SELECT scraped_comments_total, scraped_comments_top_level FROM projects WHERE project_id='p1'"
        ).fetchone()
        self.assertEqual(row["scraped_comments_total"], 2)
        self.assertEqual(row["scraped_comments_top_level"], 1)

    def test_completeness_status(self):
        upsert_project(self.conn, "p1", "https://www.kickstarter.com/projects/a/b")
        insert_comment(
            self.conn,
            {"id": "c1", "project_id": "p1", "parent_id": None, "body": "x"},
        )
        self.conn.commit()
        refresh_scraped_counts(self.conn, project_id="p1")
        update_project_ks_counts(self.conn, "p1", ks_comments_nav=1, ks_updates_nav=0)
        self.conn.commit()
        report = refresh_completeness_status(self.conn, project_id="p1")
        self.assertEqual(report[0]["comments_status"], "complete")
        self.assertEqual(report[0]["updates_status"], "expected_zero")

    def test_get_projects_needing_scrape(self):
        upsert_project(self.conn, "p1", "https://www.kickstarter.com/projects/a/b")
        update_project_ks_counts(self.conn, "p1", ks_comments_nav=5, ks_updates_nav=2)
        self.conn.execute(
            "UPDATE projects SET comments_status='partial', updates_status='missing' WHERE project_id='p1'"
        )
        self.conn.commit()
        comments_queue = get_projects_needing_scrape(self.conn, "comments")
        updates_queue = get_projects_needing_scrape(self.conn, "updates")
        self.assertEqual(len(comments_queue), 1)
        self.assertEqual(len(updates_queue), 1)

    def test_replace_comments_for_project(self):
        upsert_project(self.conn, "p1", "https://www.kickstarter.com/projects/a/b")
        replace_comments_for_project(
            self.conn,
            "p1",
            [
                {"id": "c1", "parent_id": None, "body": "a"},
                {"id": "c2", "parent_id": None, "body": "b"},
            ],
        )
        self.conn.commit()
        count = self.conn.execute("SELECT COUNT(*) FROM comments WHERE project_id='p1'").fetchone()[0]
        self.assertEqual(count, 2)
        replace_comments_for_project(self.conn, "p1", [{"id": "c3", "parent_id": None, "body": "c"}])
        self.conn.commit()
        count = self.conn.execute("SELECT COUNT(*) FROM comments WHERE project_id='p1'").fetchone()[0]
        self.assertEqual(count, 1)

    def test_insert_update_dedupes(self):
        upsert_project(self.conn, "p1", "https://www.kickstarter.com/projects/a/b")
        row = {
            "id": "u1",
            "project_id": "p1",
            "title": "Update",
            "number": 1,
            "body": "text",
        }
        self.assertTrue(insert_update(self.conn, row))
        self.assertFalse(insert_update(self.conn, row))
        self.conn.commit()


class TestConsolidate(unittest.TestCase):
    def test_ingest_comment_batch_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            comments_dir = Path(tmp) / "scraped"
            comments_dir.mkdir()
            batch = comments_dir / "kickstarter_comments_batch_1_local_20250615_120000.csv"
            pd.DataFrame(
                [
                    {
                        "id": "c1",
                        "project_id": "99",
                        "parent_id": None,
                        "body": "hi",
                        "project_slug": "a/b",
                    }
                ]
            ).to_csv(batch, index=False)

            db_path = Path(tmp) / "kickstarter.db"
            conn = connect_db(db_path)
            init_schema(conn)

            from processing.consolidate_to_sqlite import ingest_comment_batches

            inserted, skipped = ingest_comment_batches(conn, str(comments_dir))
            self.assertEqual(inserted, 1)
            self.assertEqual(skipped, 0)
            row = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
            self.assertEqual(row, 1)
            conn.close()


class TestFetchProjectCounts(unittest.TestCase):
    @patch("scrapers.fetch_project_counts.ProjectCountFetcher.fetch_page_html")
    def test_process_projects_updates_db(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_HTML
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = connect_db(db_path)
            init_schema(conn)
            df = pd.DataFrame(
                [
                    {
                        "id": "p1",
                        "project_url": "https://www.kickstarter.com/projects/a/b",
                    }
                ]
            )
            from scrapers.fetch_project_counts import ProjectCountFetcher, process_projects

            fetcher = ProjectCountFetcher()
            processed, skipped = process_projects(conn, df, "project_url", fetcher, delay=0)
            self.assertEqual(processed, 1)
            self.assertEqual(skipped, 0)
            row = conn.execute(
                "SELECT ks_comments_nav, ks_updates_nav FROM projects WHERE project_id='p1'"
            ).fetchone()
            self.assertEqual(row["ks_comments_nav"], 184)
            self.assertEqual(row["ks_updates_nav"], 0)
            conn.close()


class TestScrapeSqliteMock(unittest.TestCase):
    def test_scrape_project_stores_comments(self):
        import scrapers.scrape_comments_sqlite as comments_sqlite

        mock_scraper = MagicMock()
        mock_scraper.fetch_comments.return_value = [
            {"id": "c1", "parent_id": None, "body": "hello", "project_slug": "a/b"}
        ]

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = connect_db(db_path)
            init_schema(conn)
            upsert_project(conn, "p1", "https://www.kickstarter.com/projects/a/b")
            conn.commit()

            comments_sqlite.scrape_project(
                conn, mock_scraper, "p1", "https://www.kickstarter.com/projects/a/b"
            )
            count = conn.execute("SELECT COUNT(*) FROM comments WHERE project_id='p1'").fetchone()[0]
            self.assertEqual(count, 1)
            conn.close()

    def test_scrape_project_keeps_comments_when_blocked(self):
        import scrapers.scrape_comments_sqlite as comments_sqlite
        from scrapers.ks_session import CloudflareBlockedError

        mock_scraper = MagicMock()
        mock_scraper.fetch_comments.side_effect = CloudflareBlockedError("blocked")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conn = connect_db(db_path)
            init_schema(conn)
            upsert_project(conn, "p1", "https://www.kickstarter.com/projects/a/b")
            insert_comment(
                conn,
                {"id": "c1", "project_id": "p1", "parent_id": None, "body": "keep me"},
            )
            conn.commit()

            ok = comments_sqlite.scrape_project(
                conn, mock_scraper, "p1", "https://www.kickstarter.com/projects/a/b"
            )
            self.assertFalse(ok)
            count = conn.execute("SELECT COUNT(*) FROM comments WHERE project_id='p1'").fetchone()[0]
            self.assertEqual(count, 1)
            conn.close()


if __name__ == "__main__":
    unittest.main()
