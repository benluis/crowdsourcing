"""Unit tests for Kicktraq chart scraping. Created: 2026-07-22"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from processing.sqlite_schema import (  # noqa: E402
    connect_db,
    init_schema,
    kicktraq_charts_complete,
    upsert_kicktraq_chart,
    upsert_project,
)
from scrapers.kicktraq_parser import (  # noqa: E402
    CHART_FILENAMES,
    chart_image_url,
    kickstarter_to_kicktraq_url,
    parse_project_info_from_html,
)
from scrapers.scrape_kicktraq_charts import (  # noqa: E402
    KicktraqChartScraper,
    KicktraqRateLimitedError,
    _existing_chart_ok,
    _html_looks_ambiguous,
    charts_complete_on_disk,
    exit_code_for_results,
    is_valid_image_payload,
    scrape_project,
    should_skip_project,
    soft_backoff_seconds,
)

SAMPLE_HTML = """
<html><body>
<div id="project-infobox">
  <div class="ribbon-inner rblue shadow"><h3>Status: Active</h3></div>
  <div id="project-info">
    <a id="button-backthis"
       href="https://www.kickstarter.com/projects/foo/bar/?ref=kicktraq">
      Visit Project
    </a>
    <div id="project-info-text">
      Example description
      <div class="project-cat"><a href="/categories/games/tabletop games/">Tabletop Games</a></div>
      Backers: 36
      <br/>Average Daily Pledges: £2,632
      <br/>Average Pledge Per Backer: £73
      <br/><br/>Funding: £2,632 of £3,000
      <br/>Dates:
      <a class="datelink" title="Wednesday 22nd of July 2026 @ 12:36:32 AM (CST)">Jul 22nd</a>
      -&gt;
      <a class="datelink" title="Wednesday 5th of August 2026 @ 12:36:32 AM (CST)">Aug 5th</a>
      (14 days)
      <br/>Project By:
      <a href="https://www.kickstarter.com/projects/foo/bar/creator_bio">Colin Patten</a>
      <div style="margin-top: 10px;"><h6>Tags:</h6></div>
    </div>
  </div>
</div>
</body></html>
"""

VALID_PNG = b"\x89PNG" + (b"\x00" * 2000)
HTML_BYTES = b"<!DOCTYPE html><html><body>not a png</body></html>" + (b" " * 2000)
KS_URL = "https://www.kickstarter.com/projects/foo/bar"


def _write_valid_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(VALID_PNG)


def _seed_db_charts(conn, project_id: str, charts_dir: Path, slug: str = "foo/bar") -> None:
    upsert_project(conn, project_id, f"https://www.kickstarter.com/projects/{slug}")
    out = charts_dir / slug.replace("/", "__")
    for chart_type, filename in CHART_FILENAMES.items():
        path = out / filename
        _write_valid_png(path)
        upsert_kicktraq_chart(
            conn,
            project_id,
            chart_type,
            source_url=f"https://example.com/{filename}",
            file_path=str(path.as_posix()),
            file_size=path.stat().st_size,
            content_type="image/png",
        )
    conn.commit()


class KicktraqParserTests(unittest.TestCase):
    def test_kickstarter_to_kicktraq_url(self):
        ks = "https://www.kickstarter.com/projects/foo/bar"
        self.assertEqual(
            kickstarter_to_kicktraq_url(ks),
            "https://www.kicktraq.com/projects/foo/bar/",
        )

    def test_chart_image_url(self):
        base = "https://www.kicktraq.com/projects/foo/bar/"
        self.assertEqual(
            chart_image_url(base, "daily_pledges"),
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
        )

    def test_parse_project_info_from_html(self):
        info = parse_project_info_from_html(
            SAMPLE_HTML,
            "https://www.kicktraq.com/projects/foo/bar/",
        )
        self.assertEqual(info.backers, 36)
        self.assertEqual(info.funding_current, 2632.0)
        self.assertEqual(info.funding_goal, 3000.0)
        self.assertEqual(info.funding_currency, "£")
        self.assertEqual(info.start_date, "2026-07-22")
        self.assertEqual(info.end_date, "2026-08-05")
        self.assertEqual(info.campaign_days, 14)
        self.assertEqual(info.category, "Tabletop Games")
        self.assertEqual(info.creator_name, "Colin Patten")
        self.assertEqual(
            info.kickstarter_url,
            "https://www.kickstarter.com/projects/foo/bar/",
        )


class KicktraqSchemaTests(unittest.TestCase):
    def test_kicktraq_charts_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect_db(Path(tmp) / "test.db")
            init_schema(conn)
            upsert_project(conn, "p1", "https://www.kickstarter.com/projects/a/b")
            self.assertFalse(kicktraq_charts_complete(conn, "p1"))
            for chart_type in ("daily_pledges", "daily_backers", "daily_comments"):
                upsert_kicktraq_chart(
                    conn,
                    "p1",
                    chart_type,
                    source_url=f"https://example.com/{chart_type}.png",
                    file_path=f"data/{chart_type}.png",
                    file_size=1234,
                    content_type="image/png",
                )
            self.assertTrue(kicktraq_charts_complete(conn, "p1"))
            conn.close()

    def test_should_skip_requires_disk_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            charts_dir = tmp_path / "charts"
            upsert_project(conn, "p1", KS_URL)
            self.assertFalse(
                should_skip_project(
                    conn, "p1", force=False, charts_dir=charts_dir, kickstarter_url=KS_URL
                )
            )
            # DB-only complete is not enough when charts_dir is provided
            for chart_type in CHART_FILENAMES:
                upsert_kicktraq_chart(
                    conn,
                    "p1",
                    chart_type,
                    source_url=f"https://example.com/{chart_type}.png",
                    file_path=f"data/{chart_type}.png",
                    file_size=2000,
                    content_type="image/png",
                )
            self.assertTrue(kicktraq_charts_complete(conn, "p1"))
            self.assertFalse(
                should_skip_project(
                    conn, "p1", force=False, charts_dir=charts_dir, kickstarter_url=KS_URL
                )
            )
            _seed_db_charts(conn, "p1", charts_dir)
            self.assertTrue(
                should_skip_project(
                    conn, "p1", force=False, charts_dir=charts_dir, kickstarter_url=KS_URL
                )
            )
            self.assertFalse(
                should_skip_project(
                    conn, "p1", force=True, charts_dir=charts_dir, kickstarter_url=KS_URL
                )
            )
            conn.close()


class KicktraqImageValidationTests(unittest.TestCase):
    def test_requires_png_magic_even_with_image_content_type(self):
        self.assertTrue(is_valid_image_payload(VALID_PNG, "image/png"))
        # Forged image/* + non-PNG body must be rejected
        self.assertFalse(is_valid_image_payload(b"x" * 2000, "image/jpeg"))
        self.assertFalse(is_valid_image_payload(HTML_BYTES, "image/png"))

    def test_accepts_png_magic_without_content_type(self):
        self.assertTrue(is_valid_image_payload(VALID_PNG, ""))
        self.assertTrue(is_valid_image_payload(VALID_PNG, "application/octet-stream"))

    def test_rejects_html_and_url_extension_alone(self):
        self.assertFalse(is_valid_image_payload(HTML_BYTES, "text/html"))
        self.assertFalse(is_valid_image_payload(HTML_BYTES, "application/octet-stream"))
        # URL ending in .png must not matter — payload is what we validate
        self.assertFalse(is_valid_image_payload(b"not-png" + b"\x00" * 2000, ""))

    def test_rejects_forged_image_content_type_with_html_body(self):
        """Blocking: Content-Type image/png alone must not accept HTML."""
        self.assertFalse(is_valid_image_payload(HTML_BYTES, "image/png"))
        self.assertFalse(is_valid_image_payload(HTML_BYTES, "image/jpeg; charset=utf-8"))

    def test_existing_chart_ok_requires_png_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.png"
            bad = Path(tmp) / "bad.png"
            _write_valid_png(good)
            bad.write_bytes(HTML_BYTES)
            self.assertTrue(_existing_chart_ok(good))
            self.assertFalse(_existing_chart_ok(bad))
            self.assertFalse(_existing_chart_ok(Path(tmp) / "missing.png"))


class KicktraqAmbiguousParseTests(unittest.TestCase):
    def test_long_page_without_project_info_is_ambiguous(self):
        # Length alone must not make a page permanent — soft interstitial / empty shell
        long_shell = "<html><body>" + ("x" * 5000) + "</body></html>"
        self.assertTrue(_html_looks_ambiguous(long_shell))

    def test_page_with_project_info_not_ambiguous(self):
        self.assertFalse(_html_looks_ambiguous(SAMPLE_HTML))


class KicktraqExitCodeTests(unittest.TestCase):
    def test_exit_code_for_results(self):
        self.assertEqual(exit_code_for_results(0, 0), 0)
        self.assertEqual(exit_code_for_results(3, 0), 1)
        self.assertEqual(exit_code_for_results(1, 5), 1)  # remaining wins
        self.assertEqual(exit_code_for_results(0, 2), 2)


class KicktraqSoftRateLimitTests(unittest.TestCase):
    def test_soft_backoff_seconds(self):
        self.assertEqual(soft_backoff_seconds(0), 30.0)
        self.assertEqual(soft_backoff_seconds(1), 60.0)
        self.assertEqual(soft_backoff_seconds(2), 120.0)
        self.assertEqual(soft_backoff_seconds(3), 240.0)
        self.assertEqual(soft_backoff_seconds(4), 240.0)

    def _mock_response(
        self,
        status_code: int,
        *,
        text: str = "",
        content: bytes = b"",
        content_type: str = "text/html",
    ):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        resp.content = content
        resp.headers = {"content-type": content_type}
        return resp

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_text_retries_http_400_then_raises(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(400, text="Bad Request")

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        with self.assertRaises(KicktraqRateLimitedError) as ctx:
            scraper.fetch_text(
                "https://www.kicktraq.com/projects/foo/bar/", max_retries=3
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(session.get.call_count, 3)
        self.assertIn("HTTP 400", str(ctx.exception))

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_text_succeeds_after_400(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.side_effect = [
            self._mock_response(400, text="Bad Request"),
            self._mock_response(200, text=SAMPLE_HTML),
        ]

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        html = scraper.fetch_text("https://www.kicktraq.com/projects/foo/bar/")
        self.assertIn("project-info-text", html)
        self.assertEqual(session.get.call_count, 2)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_text_http_404_no_retry(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(404, text="Not Found")

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_text("https://www.kicktraq.com/projects/foo/bar/")
        self.assertIsNone(result)
        self.assertEqual(scraper.last_status, 404)
        self.assertEqual(session.get.call_count, 1)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_text_retries_http_503(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.side_effect = [
            self._mock_response(503, text="Unavailable"),
            self._mock_response(200, text=SAMPLE_HTML),
        ]

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        html = scraper.fetch_text("https://www.kicktraq.com/projects/foo/bar/")
        self.assertIn("project-info-text", html)
        self.assertEqual(session.get.call_count, 2)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_sends_referer_and_accept(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(
            200, content=VALID_PNG, content_type="image/png"
        )

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        referer = "https://www.kicktraq.com/projects/foo/bar/"
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer=referer,
        )
        self.assertIsNotNone(result)
        kwargs = session.get.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Referer"], referer)
        self.assertIn("image/png", kwargs["headers"]["Accept"])

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_http_404_does_not_retry_as_rate_limit(
        self, mock_create, _mock_sleep
    ):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(404, text="Not Found")

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
        )
        self.assertIsNone(result)
        self.assertEqual(session.get.call_count, 1)
        self.assertIn("HTTP 404", scraper.last_error)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_rejects_html_despite_png_url(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(
            200, content=HTML_BYTES, content_type="text/html"
        )

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
            max_retries=3,
        )
        self.assertIsNone(result)
        # Soft-block HTML is retried, then rejected
        self.assertEqual(session.get.call_count, 3)
        self.assertIn("Failed to fetch image", scraper.last_error)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_rejects_forged_image_content_type_html(
        self, mock_create, _mock_sleep
    ):
        """Forged Content-Type: image/png + HTML body must never be accepted."""
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(
            200, content=HTML_BYTES, content_type="image/png"
        )

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
            max_retries=2,
        )
        self.assertIsNone(result)
        self.assertGreaterEqual(session.get.call_count, 1)
        # Must not return HTML as a valid image
        if result is not None:
            self.fail("forged image/png + HTML must be rejected")

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_soft_retries_html_then_succeeds(
        self, mock_create, _mock_sleep
    ):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.side_effect = [
            self._mock_response(200, content=HTML_BYTES, content_type="image/png"),
            self._mock_response(200, content=VALID_PNG, content_type="image/png"),
        ]

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], VALID_PNG)
        self.assertEqual(session.get.call_count, 2)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_retries_http_503(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.side_effect = [
            self._mock_response(503, text="Unavailable", content_type="text/html"),
            self._mock_response(200, content=VALID_PNG, content_type="image/png"),
        ]

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], VALID_PNG)
        self.assertEqual(session.get.call_count, 2)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_http_410_no_retry(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(410, text="Gone")

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
        )
        self.assertIsNone(result)
        self.assertEqual(scraper.last_status, 410)
        self.assertEqual(session.get.call_count, 1)
        self.assertIn("HTTP 410", scraper.last_error)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_deadline_stops_further_requests(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(200, text=SAMPLE_HTML)

        # Deadline already expired — pause must abort before HTTP
        scraper = KicktraqChartScraper(
            request_delay=0, reset_interval=0, deadline=0.0
        )
        result = scraper.fetch_text("https://www.kicktraq.com/projects/foo/bar/")
        self.assertIsNone(result)
        self.assertEqual(session.get.call_count, 0)
        self.assertIn("Deadline", scraper.last_error)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_pause_aborts_when_sleep_capped_exhausts_deadline(
        self, mock_create, _mock_sleep
    ):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(200, text=SAMPLE_HTML)

        # Future deadline so pre-check passes; sleep_capped False aborts before GET
        scraper = KicktraqChartScraper(
            request_delay=10.0, reset_interval=0, deadline=time.time() + 3600
        )
        with patch(
            "scrapers.scrape_kicktraq_charts.sleep_capped", return_value=False
        ):
            result = scraper.fetch_text(
                "https://www.kicktraq.com/projects/foo/bar/"
            )
        self.assertIsNone(result)
        self.assertEqual(session.get.call_count, 0)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_fetch_binary_accepts_png_magic_without_image_content_type(
        self, mock_create, _mock_sleep
    ):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        session.get.return_value = self._mock_response(
            200, content=VALID_PNG, content_type="application/octet-stream"
        )

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        result = scraper.fetch_binary(
            "https://www.kicktraq.com/projects/foo/bar/dailypledges.png",
            referer="https://www.kicktraq.com/projects/foo/bar/",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], VALID_PNG)

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_reset_session_closes_previous(self, mock_create, _mock_sleep):
        old_session = MagicMock()
        new_session = MagicMock()
        mock_create.side_effect = [(old_session, "mock"), (new_session, "mock")]

        scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
        self.assertIs(scraper.session, old_session)
        scraper.reset_session()
        old_session.close.assert_called_once()
        self.assertIs(scraper.session, new_session)


class KicktraqScrapeProjectTests(unittest.TestCase):
    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_http_404_returns_permanent(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "Not Found"
        resp.headers = {"content-type": "text/html"}
        session.get.return_value = resp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            outcome = scrape_project(
                conn,
                scraper,
                "p1",
                KS_URL,
                tmp_path / "charts",
            )
            self.assertEqual(outcome, "permanent")
            self.assertEqual(session.get.call_count, 1)
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_http_410_returns_permanent(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        resp = MagicMock()
        resp.status_code = 410
        resp.text = "Gone"
        resp.content = b"Gone"
        resp.headers = {"content-type": "text/html"}
        session.get.return_value = resp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            outcome = scrape_project(
                conn,
                scraper,
                "p1",
                KS_URL,
                tmp_path / "charts",
            )
            self.assertEqual(outcome, "permanent")
            self.assertEqual(scraper.last_status, 410)
            self.assertEqual(session.get.call_count, 1)
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_chart_404_returns_permanent_not_requeued(
        self, mock_create, _mock_sleep
    ):
        """Page 200 but all chart assets 404 → permanent (no infinite requeue)."""
        session = MagicMock()
        mock_create.return_value = (session, "mock")

        page = MagicMock()
        page.status_code = 200
        page.text = SAMPLE_HTML
        page.headers = {"content-type": "text/html"}
        not_found = MagicMock()
        not_found.status_code = 404
        not_found.text = "Not Found"
        not_found.content = b"Not Found"
        not_found.headers = {"content-type": "text/html"}
        session.get.side_effect = [page, not_found, not_found, not_found]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            outcome = scrape_project(
                conn,
                scraper,
                "p1",
                KS_URL,
                tmp_path / "charts",
            )
            self.assertEqual(outcome, "permanent")
            self.assertEqual(session.get.call_count, 4)  # page + 3 charts
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_partial_chart_404_still_permanent(self, mock_create, _mock_sleep):
        """Some charts succeed, missing ones 404 → permanent incomplete."""
        session = MagicMock()
        mock_create.return_value = (session, "mock")

        page = MagicMock()
        page.status_code = 200
        page.text = SAMPLE_HTML
        page.headers = {"content-type": "text/html"}
        good_img = MagicMock()
        good_img.status_code = 200
        good_img.content = VALID_PNG
        good_img.headers = {"content-type": "image/png"}
        not_found = MagicMock()
        not_found.status_code = 404
        not_found.text = "Not Found"
        not_found.content = b"Not Found"
        not_found.headers = {"content-type": "text/html"}
        # page + pledges OK + backers 404 + comments 404
        session.get.side_effect = [page, good_img, not_found, not_found]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            outcome = scrape_project(
                conn,
                scraper,
                "p1",
                KS_URL,
                tmp_path / "charts",
            )
            self.assertEqual(outcome, "permanent")
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_ambiguous_parse_returns_retry(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>Just a moment... cf-browser-verification</body></html>"
        resp.headers = {"content-type": "text/html"}
        session.get.return_value = resp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            # Force past Cloudflare detection in fetch_text by mocking it off
            with patch(
                "scrapers.scrape_kicktraq_charts.is_cloudflare_challenge",
                return_value=False,
            ):
                outcome = scrape_project(
                    conn,
                    scraper,
                    "p1",
                    KS_URL,
                    tmp_path / "charts",
                )
            self.assertEqual(outcome, "retry")
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_long_shell_without_project_info_is_retry(
        self, mock_create, _mock_sleep
    ):
        session = MagicMock()
        mock_create.return_value = (session, "mock")
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "<html><body>" + ("no project markup " * 200) + "</body></html>"
        resp.headers = {"content-type": "text/html"}
        session.get.return_value = resp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            with patch(
                "scrapers.scrape_kicktraq_charts.is_cloudflare_challenge",
                return_value=False,
            ):
                outcome = scrape_project(
                    conn,
                    scraper,
                    "p1",
                    KS_URL,
                    tmp_path / "charts",
                )
            self.assertEqual(outcome, "retry")
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    def test_preserve_on_failure_and_skip_complete(self, mock_create, _mock_sleep):
        session = MagicMock()
        mock_create.return_value = (session, "mock")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            charts_dir = tmp_path / "charts"
            conn = connect_db(tmp_path / "test.db")
            init_schema(conn)
            _seed_db_charts(conn, "p1", charts_dir)

            self.assertTrue(charts_complete_on_disk(charts_dir, KS_URL))
            self.assertTrue(
                should_skip_project(
                    conn,
                    "p1",
                    force=False,
                    charts_dir=charts_dir,
                    kickstarter_url=KS_URL,
                )
            )

            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            outcome = scrape_project(
                conn, scraper, "p1", KS_URL, charts_dir, force=False
            )
            self.assertEqual(outcome, "complete")
            self.assertEqual(session.get.call_count, 0)

            # Corrupt one file → must not skip; failed re-download preserves others
            out = charts_dir / "foo__bar"
            (out / "dailypledges.png").write_bytes(HTML_BYTES)
            # Clear DB so skip doesn't short-circuit on DB+disk
            conn.execute("DELETE FROM kicktraq_charts WHERE project_id = ?", ("p1",))
            conn.commit()

            page = MagicMock()
            page.status_code = 200
            page.text = SAMPLE_HTML
            page.headers = {"content-type": "text/html"}
            # Non-HTML invalid bytes: not soft-block, immediate reject (no soft retry)
            junk = b"\x00" * 2000
            bad_img = MagicMock()
            bad_img.status_code = 200
            bad_img.content = junk
            bad_img.text = ""
            bad_img.headers = {"content-type": "application/octet-stream"}
            # page + 1 chart attempt for corrupted pledges; backers+comments kept on disk
            session.get.side_effect = [page, bad_img]

            outcome = scrape_project(
                conn, scraper, "p1", KS_URL, charts_dir, force=False
            )
            # dailypledges fetch fails (invalid, not soft-block) → not preserved → retry
            self.assertEqual(outcome, "retry")
            self.assertTrue(_existing_chart_ok(out / "dailybackers.png"))
            self.assertTrue(_existing_chart_ok(out / "dailycomments.png"))
            conn.close()


class KicktraqChartFilenamesTests(unittest.TestCase):
    def test_three_chart_types(self):
        self.assertEqual(len(CHART_FILENAMES), 3)


class KicktraqRequeueTests(unittest.TestCase):
    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    @patch("scrapers.scrape_kicktraq_charts.scrape_project")
    def test_requeues_retry_until_complete(self, mock_scrape, mock_create, _mock_sleep):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import process_projects

        mock_create.return_value = (MagicMock(), "mock")
        mock_scrape.side_effect = ["retry", "complete"]

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            charts_dir = Path(tmp) / "charts"
            charts_dir.mkdir()
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
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            processed, skipped, permanent, remaining = process_projects(
                conn,
                df,
                "project_url",
                scraper,
                charts_dir,
                delay=0,
                block_cooldown=0,
                pass_cooldown=0,
                max_runtime_seconds=3600,
                requeue=True,
            )
            self.assertEqual(processed, 1)
            self.assertEqual(permanent, 0)
            self.assertEqual(remaining, 0)
            self.assertEqual(mock_scrape.call_count, 2)
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    @patch("scrapers.scrape_kicktraq_charts.scrape_project")
    def test_permanent_not_requeued(self, mock_scrape, mock_create, _mock_sleep):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import process_projects

        mock_create.return_value = (MagicMock(), "mock")
        mock_scrape.return_value = "permanent"

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            charts_dir = Path(tmp) / "charts"
            charts_dir.mkdir()
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
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            processed, _skipped, permanent, remaining = process_projects(
                conn,
                df,
                "project_url",
                scraper,
                charts_dir,
                delay=0,
                block_cooldown=0,
                pass_cooldown=0,
                max_runtime_seconds=3600,
                requeue=True,
            )
            self.assertEqual(processed, 0)
            self.assertEqual(permanent, 1)
            self.assertEqual(remaining, 0)
            self.assertEqual(mock_scrape.call_count, 1)
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    @patch("scrapers.scrape_kicktraq_charts.scrape_project")
    def test_unexpected_exception_requeued(self, mock_scrape, mock_create, _mock_sleep):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import process_projects

        mock_create.return_value = (MagicMock(), "mock")
        mock_scrape.side_effect = [RuntimeError("boom"), "complete"]

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            charts_dir = Path(tmp) / "charts"
            charts_dir.mkdir()
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
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            processed, _skipped, permanent, remaining = process_projects(
                conn,
                df,
                "project_url",
                scraper,
                charts_dir,
                delay=0,
                block_cooldown=0,
                pass_cooldown=0,
                max_runtime_seconds=3600,
                requeue=True,
            )
            self.assertEqual(processed, 1)
            self.assertEqual(permanent, 0)
            self.assertEqual(remaining, 0)
            self.assertEqual(mock_scrape.call_count, 2)
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    @patch("scrapers.scrape_kicktraq_charts.scrape_project")
    def test_no_requeue_single_pass(self, mock_scrape, mock_create, _mock_sleep):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import process_projects

        mock_create.return_value = (MagicMock(), "mock")
        mock_scrape.return_value = "retry"

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            charts_dir = Path(tmp) / "charts"
            charts_dir.mkdir()
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
            scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
            processed, skipped, permanent, remaining = process_projects(
                conn,
                df,
                "project_url",
                scraper,
                charts_dir,
                delay=0,
                block_cooldown=0,
                pass_cooldown=0,
                max_runtime_seconds=3600,
                requeue=False,
            )
            self.assertEqual(processed, 0)
            self.assertEqual(remaining, 1)
            self.assertEqual(mock_scrape.call_count, 1)
            conn.close()

    @patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None)
    @patch("scrapers.scrape_kicktraq_charts.create_kickstarter_session")
    @patch("scrapers.scrape_kicktraq_charts.scrape_project")
    def test_stops_at_max_runtime(self, mock_scrape, mock_create, _mock_sleep):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import process_projects

        mock_create.return_value = (MagicMock(), "mock")
        mock_scrape.return_value = "retry"

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            charts_dir = Path(tmp) / "charts"
            charts_dir.mkdir()
            conn = connect_db(db_path)
            try:
                init_schema(conn)
                df = pd.DataFrame(
                    [
                        {
                            "id": "p1",
                            "project_url": "https://www.kickstarter.com/projects/a/b",
                        }
                    ]
                )
                scraper = KicktraqChartScraper(request_delay=0, reset_interval=0)
                _processed, _skipped, _permanent, remaining = process_projects(
                    conn,
                    df,
                    "project_url",
                    scraper,
                    charts_dir,
                    delay=0,
                    block_cooldown=0,
                    pass_cooldown=0,
                    max_runtime_seconds=0,  # already expired
                    requeue=True,
                )
                self.assertEqual(remaining, 1)
                self.assertEqual(mock_scrape.call_count, 0)
            finally:
                conn.close()


class KicktraqMainExitTests(unittest.TestCase):
    def test_main_exits_nonzero_on_remaining(self):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "in.csv"
            pd.DataFrame(
                [
                    {
                        "id": "p1",
                        "project_url": "https://www.kickstarter.com/projects/a/b",
                    }
                ]
            ).to_csv(csv_path, index=False)
            db_path = tmp_path / "test.db"
            charts_dir = tmp_path / "charts"

            with (
                patch(
                    "sys.argv",
                    [
                        "scrape_kicktraq_charts.py",
                        str(csv_path),
                        "--db",
                        str(db_path),
                        "--charts-dir",
                        str(charts_dir),
                        "--delay",
                        "0",
                        "--request-delay",
                        "0",
                        "--block-cooldown",
                        "0",
                        "--pass-cooldown",
                        "0",
                        "--no-requeue",
                        "--max-runtime-hours",
                        "1",
                    ],
                ),
                patch(
                    "scrapers.scrape_kicktraq_charts.create_kickstarter_session",
                    return_value=(MagicMock(), "mock"),
                ),
                patch(
                    "scrapers.scrape_kicktraq_charts.scrape_project",
                    return_value="retry",
                ),
                patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None),
                self.assertRaises(SystemExit) as ctx,
            ):
                main()
            self.assertEqual(ctx.exception.code, 1)

    def test_main_exits_2_on_permanent_only(self):
        import pandas as pd
        from scrapers.scrape_kicktraq_charts import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "in.csv"
            pd.DataFrame(
                [
                    {
                        "id": "p1",
                        "project_url": "https://www.kickstarter.com/projects/a/b",
                    }
                ]
            ).to_csv(csv_path, index=False)
            db_path = tmp_path / "test.db"
            charts_dir = tmp_path / "charts"

            with (
                patch(
                    "sys.argv",
                    [
                        "scrape_kicktraq_charts.py",
                        str(csv_path),
                        "--db",
                        str(db_path),
                        "--charts-dir",
                        str(charts_dir),
                        "--delay",
                        "0",
                        "--request-delay",
                        "0",
                        "--block-cooldown",
                        "0",
                        "--pass-cooldown",
                        "0",
                        "--no-requeue",
                        "--max-runtime-hours",
                        "1",
                    ],
                ),
                patch(
                    "scrapers.scrape_kicktraq_charts.create_kickstarter_session",
                    return_value=(MagicMock(), "mock"),
                ),
                patch(
                    "scrapers.scrape_kicktraq_charts.scrape_project",
                    return_value="permanent",
                ),
                patch("scrapers.scrape_kicktraq_charts.time.sleep", return_value=None),
                self.assertRaises(SystemExit) as ctx,
            ):
                main()
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
