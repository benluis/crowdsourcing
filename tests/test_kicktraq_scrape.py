"""Unit tests for Kicktraq chart scraping. Created: 2026-07-22"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

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
    chart_image_url,
    kickstarter_to_kicktraq_url,
    parse_project_info_from_html,
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


if __name__ == "__main__":
    unittest.main()
