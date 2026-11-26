"""Headless UI smoke check; device components and messaging settings are mocked."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from episode_store import EpisodeStore


class AppTests(unittest.TestCase):
    def test_render_reports_and_opt_in_controls(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmp:
            store = EpisodeStore(Path(tmp) / "events.sqlite3")
            component = lambda *args, **kwargs: {
                "armed": False,
                "location": None,
                "location_status": "denied",
            }
            with patch(
                "streamlit.components.v1.declare_component", return_value=component
            ), patch("episode_store.EpisodeStore", return_value=store), patch(
                "streamlit.secrets", SimpleNamespace(to_dict=lambda: {})
            ), patch(
                "streamlit_webrtc.webrtc_streamer",
                return_value=SimpleNamespace(
                    state=SimpleNamespace(playing=False), video_processor=None
                ),
            ):
                app = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
                ).run(timeout=30)
                self.assertFalse(app.exception, list(app.exception))
                self.assertEqual(len(app.tabs), 6)
                consent = next(
                    c for c in app.checkbox if c.label.startswith("I consent")
                )
                self.assertFalse(consent.value)
                message = next(
                    c for c in app.checkbox if c.label.startswith("Enable TEST")
                )
                self.assertFalse(message.value)
                self.assertTrue(message.disabled)
                self.assertTrue(
                    any("Total episodes:" in entry.value for entry in app.markdown)
                )

                # A failed TEST send must remain visible without breaking the app/report.
                runtime_owner = app.session_state.monitor_owner + ":Driver"
                store.queue(
                    runtime_owner, "fixture-job", {"message": "TEST - fixture"}, 0
                )
                store.job_result("fixture-job", "failed", 3, 0, "mock failure")
                app.run(timeout=30)
                self.assertFalse(app.exception, list(app.exception))
                self.assertTrue(
                    any("failed" in table.value.to_string() for table in app.dataframe)
                )


if __name__ == "__main__":
    unittest.main()
