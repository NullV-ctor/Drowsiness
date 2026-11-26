"""Desktop EAR monitor sharing the browser's clock-based episode rules.

Run: python drowsiness_detect.py [--night] [--consent-screenshots]
Delete consented desktop captures: python drowsiness_detect.py --delete-screenshots
Original project: mohitwildbeast/Driver-Drowsiness-Detector; EAR: PyImageSearch.
"""

import argparse
from pathlib import Path

import cv2

from detection import FaceEyeDetector, annotate
from episode_store import EpisodeStore
from monitoring import Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night", action="store_true")
    parser.add_argument(
        "--delay", type=float, help="Override day=2.0 or night=1.5 seconds"
    )
    parser.add_argument("--driver", default="Driver")
    parser.add_argument(
        "--consent-screenshots",
        action="store_true",
        help="Explicitly consent to one JPEG per episode",
    )
    parser.add_argument("--delete-screenshots", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    store = EpisodeStore(root / ".monitor/events.sqlite3")
    owner = "desktop:" + args.driver
    if args.delete_screenshots:
        store.delete_all_screenshots(owner)
        print("Deleted this desktop driver profile's episode screenshots.")
        return
    import pygame

    pygame.mixer.init()
    alert = pygame.mixer.Sound(str(root / "audio/alert.wav"))
    detector = FaceEyeDetector()
    monitor = Monitor(
        store, owner, args.driver, delay=args.delay or (1.5 if args.night else 2.0)
    )
    monitor.configure(args.consent_screenshots, False, None)
    capture = cv2.VideoCapture(0)
    sounding = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            observation = detector.observe(frame)
            state = monitor.observe(
                observation["closed"] if observation else None,
                frame,
                box=observation["box"] if observation else None,
            )
            if state.alarm and not sounding:
                alert.play(loops=-1)
            if not state.alarm:
                alert.stop()
            sounding = state.alarm
            status = (
                "NO FACE"
                if observation is None
                else "DROWSY"
                if state.alarm
                else "EYES CLOSED"
                if observation["closed"]
                else "EYES OPEN"
            )
            cv2.imshow(
                "Video", annotate(frame, observation, status, state.closed_seconds)
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        alert.stop()
        capture.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
