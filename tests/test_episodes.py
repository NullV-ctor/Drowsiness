"""Requested episode cases, using a fake clock, mock location and mock messaging."""

import io
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from episodes import EpisodeTracker
from monitoring import Monitor
from episode_store import EpisodeStore
from emergency_alerts import (
    AlertDispatcher,
    DeliveryUnknown,
    RetryableSendError,
    location_description,
    make_message,
)
from episode_reports import report_rows, report_csv, report_pdf


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wall(self):
        return datetime(2026, 11, 26, 10, tzinfo=timezone.utc) + timedelta(
            seconds=self.now
        )


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = EpisodeStore(Path(self.folder.name) / "events.sqlite3")
        self.clock = Clock()
        self.dispatch = Mock()
        self.monitor = Monitor(
            self.store,
            "owner",
            "Driver 7",
            dispatcher=self.dispatch,
            clock=self.clock.monotonic,
            wall_clock=self.clock.wall,
        )
        self.monitor.configure(True, True, None)
        self.frame = np.zeros((32, 64, 3), np.uint8)

    def observe(self, closed, t):
        self.clock.now = t
        return self.monitor.observe(closed, self.frame)

    def close_until(self, end, start=0):
        steps = int(round((end - start) * 10))
        state = None
        for i in range(steps + 1):
            state = self.observe(True, round(start + i / 10, 5))
        return state

    def test_01_blink_no_alarm_screenshot_or_message(self):
        self.close_until(0.2)
        self.assertFalse(self.observe(False, 0.3).alarm)
        self.assertEqual(self.store.episodes("owner"), [])
        self.assertEqual(self.store.screenshots("owner"), [])
        self.dispatch.enqueue.assert_not_called()

    def test_02_day_and_night_delay_and_first_closure_timestamp(self):
        for delay in (2.0, 1.5):
            tracker = EpisodeTracker(
                delay=delay, clock=self.clock.monotonic, wall_clock=self.clock.wall
            )
            for i in range(int(delay * 10)):
                self.clock.now = i / 10
                self.assertFalse(tracker.update(True).alarm)
            self.clock.now = delay
            state = tracker.update(True)
            self.assertTrue(state.alarm)
            self.assertEqual(
                state.started.started_at,
                datetime(2026, 11, 26, 10, tzinfo=timezone.utc),
            )

    def test_03_open_stops_alarm_immediately_before_episode_ends(self):
        self.assertTrue(self.close_until(2).alarm)
        self.assertFalse(self.observe(False, 2.1).alarm)
        self.assertIsNotNone(self.monitor.tracker.active)
        self.assertIsNotNone(self.observe(False, 2.6).ended)

    def test_04_thirty_seconds_one_screenshot_episode_message(self):
        self.close_until(30)
        self.assertEqual(len(self.store.episodes("owner")), 1)
        self.assertEqual(len(self.store.screenshots("owner")), 1)
        self.dispatch.enqueue.assert_called_once()
        self.assertIn("TEST", self.dispatch.enqueue.call_args.args[2])

    def test_05_under_ten_no_message(self):
        self.close_until(9.9)
        self.dispatch.enqueue.assert_not_called()

    def test_06_at_ten_exactly_one_message_with_location(self):
        fix = {
            "latitude": 28.6,
            "longitude": 77.2,
            "accuracy": 12.5,
            "timestamp": self.clock.wall().timestamp(),
        }
        self.monitor.configure(True, True, fix)
        self.close_until(10)
        self.dispatch.enqueue.assert_called_once()
        message = self.dispatch.enqueue.call_args.args[2]
        for text in (
            "TEST",
            "Driver 7",
            "10.0 seconds",
            "google.com/maps",
            "12.5 m",
            "2026-11-26",
        ):
            self.assertIn(text, message)
        self.close_until(11, start=10.1)
        self.dispatch.enqueue.assert_called_once()

    def test_07_two_separate_episodes_two_screenshots(self):
        self.close_until(2)
        self.observe(False, 2.1)
        self.observe(False, 2.6)
        self.close_until(5, start=3)
        self.assertEqual(len(self.store.episodes("owner")), 2)
        self.assertEqual(len(self.store.screenshots("owner")), 2)

    def test_two_emergency_episodes_each_get_one_job(self):
        self.close_until(10)
        self.observe(False, 10.1)
        self.observe(False, 10.6)
        self.close_until(21, start=11)
        self.assertEqual(self.dispatch.enqueue.call_count, 2)
        self.assertEqual(len(self.store.episodes("owner")), 2)
        self.assertEqual(len(self.store.screenshots("owner")), 2)

    def test_08_flicker_same_episode_but_emergency_clock_resets(self):
        self.close_until(9)
        self.observe(False, 9.1)
        self.assertTrue(self.observe(True, 9.4).alarm)
        self.close_until(18, start=9.5)
        self.dispatch.enqueue.assert_not_called()
        self.close_until(19.4, start=18.1)
        self.dispatch.enqueue.assert_called_once()
        self.assertEqual(len(self.store.episodes("owner")), 1)
        self.assertEqual(len(self.store.screenshots("owner")), 1)

    def test_09_no_face_does_not_count_as_closure(self):
        self.close_until(1.9)
        self.assertFalse(self.observe(None, 2).alarm)
        self.assertTrue(self.monitor.latest.warning)
        self.assertFalse(self.observe(True, 2.1).alarm)
        self.assertEqual(self.store.episodes("owner"), [])
        self.close_until(4.1, start=2.2)
        self.assertTrue(self.observe(None, 4.2).warning)
        self.assertFalse(self.monitor.latest.alarm)

    def test_10_denied_location_does_not_block_message_or_video(self):
        self.monitor.configure(True, True, {"status": "denied"})
        self.close_until(10)
        self.assertIn("Location unavailable", self.dispatch.enqueue.call_args.args[2])
        self.assertFalse(self.observe(False, 10.1).alarm)

    def test_11_report_only_date_slots_count_and_correct_end(self):
        self.close_until(2)
        self.observe(False, 2.1)
        self.observe(False, 2.6)
        rows = report_rows(self.store.episodes("owner"), date(2026, 11, 26), "UTC")
        self.assertEqual(
            rows,
            [{"Start": "2026-11-26 10:00:00.000", "End": "2026-11-26 10:00:02.600"}],
        )
        csv = report_csv(date(2026, 11, 26), rows).decode()
        self.assertIn("Date,Start,End,Total episodes", csv)
        self.assertNotIn("screenshot", csv.lower())
        pdf = report_pdf(date(2026, 11, 26), rows)
        self.assertTrue(pdf.startswith(b"%PDF"))
        from pypdf import PdfReader

        doc = PdfReader(io.BytesIO(pdf))
        text = "\n".join(p.extract_text() for p in doc.pages)
        for required in ("2026-11-26", "Total episodes: 1", "10:00:00", "10:00:02"):
            self.assertIn(required, text)
        for banned in ("Driver", "advice", "chart", "Screenshot", "Accuracy", "EAR"):
            self.assertNotIn(banned, text)
        self.assertFalse(any(p.images for p in doc.pages))

    def test_no_consent_no_image_and_delete_does_not_recapture(self):
        self.monitor.configure(False, False, None)
        self.close_until(2)
        self.assertEqual(self.store.screenshots("owner"), [])
        self.observe(False, 2.1)
        self.observe(False, 2.6)
        self.monitor.configure(True, False, None)
        self.close_until(5, start=3)
        shot = self.store.screenshots("owner")[0]
        self.store.delete_screenshot("other", shot["id"])
        self.assertEqual(len(self.store.screenshots("owner")), 1)
        self.store.delete_screenshot("owner", shot["id"])
        self.close_until(6, start=5.1)
        self.assertEqual(self.store.screenshots("owner"), [])
        self.assertEqual(len(self.store.episodes("owner")), 2)

    def test_gap_cannot_satisfy_emergency_continuity(self):
        self.close_until(9)
        restarted = self.observe(True, 20)
        self.assertEqual(restarted.closed_seconds, 0)
        self.assertFalse(restarted.alarm)
        self.dispatch.enqueue.assert_not_called()


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = EpisodeStore(Path(self.folder.name) / "events.sqlite3")
        self.clock = Clock()

    def test_retry_failure_status_and_one_logical_job(self):
        sender = Mock(side_effect=[RetryableSendError("offline"), None])
        dispatcher = AlertDispatcher(self.store, sender, clock=self.clock.monotonic)
        self.store.queue("owner", "episode", {"message": "TEST"}, 0)
        self.store.queue("owner", "episode", {"message": "TEST"}, 0)
        dispatcher.process_one()
        self.assertEqual(self.store.jobs("owner")[0]["status"], "retrying")
        self.assertFalse(dispatcher.process_one())
        self.clock.now = 2
        dispatcher.process_one()
        self.assertEqual(self.store.jobs("owner")[0]["status"], "sent")
        self.assertEqual(sender.call_count, 2)
        self.assertEqual(len(self.store.jobs("owner")), 1)

    def test_uncertain_delivery_is_not_automatically_duplicated(self):
        sender = Mock(side_effect=DeliveryUnknown("uncertain"))
        dispatcher = AlertDispatcher(self.store, sender, clock=self.clock.monotonic)
        self.store.queue("owner", "episode", {"message": "TEST"}, 0)
        dispatcher.process_one()
        self.clock.now = 20
        self.assertFalse(dispatcher.process_one())
        self.assertEqual(self.store.jobs("owner")[0]["status"], "delivery_unknown")
        sender.assert_called_once()

    def test_slow_sender_does_not_block_enqueue(self):
        entered, release = threading.Event(), threading.Event()

        def slow(message):
            entered.set()
            release.wait(3)

        dispatcher = AlertDispatcher(self.store, slow)
        self.addCleanup(dispatcher.close)
        dispatcher.enqueue("owner", "episode", "TEST")
        self.assertTrue(entered.wait(1))
        # The caller is free to process another observation while the worker is blocked.
        tracker = EpisodeTracker()
        self.assertFalse(tracker.update(None).alarm)
        release.set()
        dispatcher.thread.join(
            0.1
        )  # Worker remains a daemon, processing later retry jobs if needed.

    def test_telegram_transport_is_mocked_and_only_allows_test_messages(self):
        from urllib.parse import parse_qs
        from emergency_alerts import TelegramSender

        sender = TelegramSender("SECRET_TEST_TOKEN", "configured-chat")
        with self.assertRaisesRegex(ValueError, "Only TEST"):
            sender("unlabeled message")
        response = io.BytesIO(b'{"ok": true}')
        from unittest.mock import patch

        with patch(
            "emergency_alerts.request.urlopen", return_value=response
        ) as network:
            sender("TEST - test fixture")
            request = network.call_args.args[0]
            body = parse_qs(request.data.decode())
            self.assertEqual(body["chat_id"], ["configured-chat"])
            self.assertEqual(body["text"], ["TEST - test fixture"])

    def test_transport_rate_limit_backoff_and_secret_redaction(self):
        from urllib.error import HTTPError
        from unittest.mock import patch
        from emergency_alerts import TelegramSender

        sender = TelegramSender("SECRET_TEST_TOKEN", "configured-chat")
        error = HTTPError(
            "https://example.invalid/SECRET_TEST_TOKEN",
            429,
            "limited",
            {},
            io.BytesIO(b'{"parameters":{"retry_after":8}}'),
        )
        with patch("emergency_alerts.request.urlopen", side_effect=error):
            with self.assertRaises(RetryableSendError) as caught:
                sender("TEST - test fixture")
            self.assertEqual(caught.exception.retry_after, 8)
            self.assertNotIn("SECRET_TEST_TOKEN", str(caught.exception))

    def test_stale_and_invalid_location_are_unavailable(self):
        for fix in (
            None,
            {"latitude": 999, "longitude": 0, "accuracy": 1, "timestamp": 0},
            {"latitude": 1, "longitude": 2, "accuracy": 3, "timestamp": 0},
        ):
            self.assertIn("unavailable", location_description(fix, 1000))


if __name__ == "__main__":
    unittest.main()
