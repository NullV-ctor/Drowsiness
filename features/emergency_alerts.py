"""TEST-only Telegram transport and asynchronous, one-job-per-episode delivery."""

import json
import math
import threading
import time
from urllib import error, parse, request


class RetryableSendError(Exception):
    def __init__(self, message, retry_after=0):
        super().__init__(message)
        self.retry_after = retry_after


class DeliveryUnknown(Exception):
    pass


def location_description(location, now, max_age=60):
    try:
        lat, lon, accuracy = (
            float(location[key]) for key in ("latitude", "longitude", "accuracy")
        )
        age = now - float(location["timestamp"])
        if not all(math.isfinite(value) for value in (lat, lon, accuracy, age)):
            raise ValueError
        if not (
            -90 <= lat <= 90
            and -180 <= lon <= 180
            and accuracy >= 0
            and -5 <= age <= max_age
        ):
            raise ValueError
        return f"Location: https://www.google.com/maps?q={lat:.6f},{lon:.6f}\nAccuracy: +/-{accuracy:.1f} m"
    except (TypeError, ValueError, KeyError):
        return "Location unavailable (permission denied, not received, invalid, or older than 60 seconds)."


def make_message(driver, episode, duration, location, wall_time):
    return (
        f"TEST - Driver drowsiness alert\nEpisode: {episode.id}\n"
        f"Driver: {driver}\nTime: {wall_time.isoformat()}\n"
        f"Continuous closed-eye duration: {duration:.1f} seconds\n"
        + location_description(location, wall_time.timestamp())
    )


class TelegramSender:
    def __init__(self, token, chat_id):
        if not token or not chat_id:
            raise ValueError(
                "Configure a Telegram bot token and TEST chat recipient in Streamlit secrets."
            )
        self.token, self.chat_id = token, str(chat_id)

    def __call__(self, message):
        if not message.startswith("TEST - "):
            raise ValueError("Only TEST messages are allowed by this transport.")
        body = parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": message,
                "disable_web_page_preview": "true",
            }
        ).encode()
        req = request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data=body,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=8) as response:
                result = json.load(response)
            if not result.get("ok"):
                raise RetryableSendError("Telegram rejected the request.")
        except error.HTTPError as exc:
            # Never expose exception text containing the token-bearing URL.
            if exc.code == 429 or exc.code >= 500:
                retry_after = 0
                if exc.code == 429:
                    try:
                        retry_after = max(
                            0,
                            float(
                                json.loads(exc.read(4096))
                                .get("parameters", {})
                                .get("retry_after", 0)
                            ),
                        )
                    except (ValueError, TypeError):
                        pass
                raise RetryableSendError(
                    f"Telegram HTTP {exc.code}", retry_after=retry_after
                ) from None
            raise ValueError(
                f"Telegram configuration/recipient rejected (HTTP {exc.code})."
            ) from None
        except (TimeoutError, error.URLError, ConnectionError):
            # Could have been accepted before the connection failed. Avoid automatic duplicate sends.
            raise DeliveryUnknown(
                "Delivery uncertain after network failure; inspect the TEST chat before retrying."
            ) from None


class AlertDispatcher:
    def __init__(self, store, sender, clock=time.time, max_attempts=3):
        self.store, self.sender, self.clock = store, sender, clock
        self.max_attempts = max_attempts
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = None
        self.stop_event = threading.Event()

    def enqueue(self, owner, episode_id, message):
        self.store.queue(owner, episode_id, {"message": message}, self.clock())
        self.wake.set()
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(
                    target=self._run, daemon=True, name="test-alert-delivery"
                )
                self.thread.start()

    def process_one(self):
        """One deterministic worker step, also used by fake-clock tests."""
        row = self.store.due_job(self.clock())
        if row is None:
            return False
        attempts = row["attempts"] + 1
        self.store.job_result(row["episode_id"], "sending", attempts, self.clock())
        try:
            self.sender(json.loads(row["payload"])["message"])
        except DeliveryUnknown as exc:
            self.store.job_result(
                row["episode_id"], "delivery_unknown", attempts, self.clock(), str(exc)
            )
        except RetryableSendError as exc:
            status = "retrying" if attempts < self.max_attempts else "failed"
            self.store.job_result(
                row["episode_id"],
                status,
                attempts,
                self.clock() + max(2**attempts, exc.retry_after),
                str(exc),
            )
        except Exception:
            self.store.job_result(
                row["episode_id"],
                "failed",
                attempts,
                self.clock(),
                "Configuration or send failure; inspect secrets and TEST recipient.",
            )
        else:
            self.store.job_result(row["episode_id"], "sent", attempts, self.clock())
        return True

    def _run(self):
        idle = 0
        while idle < 120 and not self.stop_event.is_set():
            if self.process_one():
                idle = 0
                continue
            self.wake.wait(0.5)
            self.wake.clear()
            idle = 0 if self.store.has_pending() else idle + 0.5

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)
