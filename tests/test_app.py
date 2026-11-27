"""Headless UI smoke check; device components and messaging settings are mocked."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from episode_store import EpisodeStore


class AppTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st

        st.cache_resource.clear()

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
                workspace = next(r for r in app.radio if r.label == "Workspace")
                self.assertEqual(len(workspace.options), 6)
                consent = next(
                    c for c in app.checkbox if c.label.startswith("I consent")
                )
                self.assertFalse(consent.value)
                message = next(
                    c for c in app.checkbox if c.label.startswith("Enable TEST")
                )
                self.assertFalse(message.value)
                self.assertTrue(message.disabled)
                consent.check().run()
                self.assertTrue(
                    next(
                        c for c in app.checkbox if c.label.startswith("Enable TEST")
                    ).disabled
                )
                workspace.set_value("Reports").run()
                self.assertFalse(app.exception, list(app.exception))
                self.assertTrue(
                    any("Total episodes:" in entry.value for entry in app.markdown)
                )

                # A failed TEST send must remain visible without breaking the app/report.
                runtime_owner = app.session_state.monitor_owner + ":Driver"
                store.queue(
                    runtime_owner, "fixture-job", {"message": "TEST - fixture"}, 0
                )
                store.job_result("fixture-job", "failed", 3, 0, "mock failure")
                next(r for r in app.radio if r.label == "Workspace").set_value(
                    "Live webcam"
                ).run(timeout=30)
                self.assertFalse(app.exception, list(app.exception))
                self.assertTrue(
                    any("failed" in table.value.to_string() for table in app.dataframe)
                )

                for page in [
                    "Image / snapshot",
                    "Video analysis",
                    "Consented screenshots",
                    "Model insights",
                ]:
                    next(r for r in app.radio if r.label == "Workspace").set_value(
                        page
                    ).run(timeout=30)
                    self.assertFalse(app.exception, list(app.exception))
                next(r for r in app.radio if r.label == "Workspace").set_value(
                    "Video analysis"
                ).run()
                self.assertTrue(
                    next(b for b in app.button if b.label == "Analyze video").disabled
                )

    def test_configured_messaging_is_independent_of_screenshot_consent(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "streamlit.components.v1.declare_component",
                return_value=lambda **kwargs: {},
            ), patch(
                "episode_store.EpisodeStore",
                return_value=EpisodeStore(Path(tmp) / "events.sqlite3"),
            ), patch(
                "streamlit.secrets",
                SimpleNamespace(
                    to_dict=lambda: {
                        "emergency": {
                            "bot_token": "fixture-token",
                            "chat_id": "fixture-chat",
                        }
                    }
                ),
            ):
                app = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
                ).run(timeout=30)
                self.assertFalse(app.exception, list(app.exception))
                self.assertFalse(
                    next(
                        c for c in app.checkbox if c.label.startswith("I consent")
                    ).value
                )
                message = next(
                    c for c in app.checkbox if c.label.startswith("Enable TEST")
                )
                self.assertFalse(message.disabled)
                message.check().run()
                self.assertFalse(app.exception, list(app.exception))
                self.assertFalse(
                    next(
                        c for c in app.checkbox if c.label.startswith("I consent")
                    ).value
                )


if __name__ == "__main__":
    unittest.main()
