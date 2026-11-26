"""Run with: streamlit run streamlit_app.py"""

import csv
import hashlib
import io
import json
import tempfile
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import streamlit as st

from detection import (
    MODEL_PATH,
    ROOT,
    ClosureTracker,
    FaceEyeDetector,
    analyze_frame,
    annotate,
)
from episode_store import EpisodeStore
from episodes import EpisodeTracker
from monitoring import Monitor
from emergency_alerts import AlertDispatcher, TelegramSender
from episode_reports import report_rows, report_csv, report_pdf
from streamlit.components.v1 import declare_component

browser_controls = declare_component(
    "driver_browser_controls", path=str(ROOT / "browser_controls")
)


@st.cache_resource
def episode_store():
    return EpisodeStore(ROOT / ".monitor" / "events.sqlite3")


@st.cache_resource
def dispatcher_for(token, chat_id):
    return AlertDispatcher(episode_store(), TelegramSender(token, chat_id))


try:
    settings = st.secrets.to_dict()
except Exception:
    # Do not expose secret values through parser/network error tracebacks.
    settings = {}
store = episode_store()
if "monitor_owner" not in st.session_state:
    st.session_state.monitor_owner = uuid.uuid4().hex
playing = st.session_state.get("stream_playing", False)

st.set_page_config(page_title="Drowsiness Monitor", page_icon="👁️", layout="wide")


@st.cache_resource
def load_detector(model_path, model_modified):
    # Timestamp invalidates the cached classifier after notebook training.
    return FaceEyeDetector(model_path=model_path)


def csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


st.title("Drowsiness Monitor")
st.caption("Train your eye-state model in the notebook, then test it here.")
with st.sidebar:
    st.header("Detection settings")
    trained = MODEL_PATH.is_file()
    method = st.selectbox(
        "Detection method",
        ["Trained eye classifier", "EAR threshold"],
        index=0 if trained else 1,
        disabled=playing,
    )
    driver = st.text_input(
        "Driver name or ID", "Driver", max_chars=80, disabled=playing
    )
    mode = st.radio("Mode", ["Day", "Night"], horizontal=True, disabled=playing)
    day_delay = st.slider(
        "Day alert delay (seconds)", 0.5, 5.0, 2.0, 0.1, disabled=playing
    )
    night_delay = st.slider(
        "Night alert delay (seconds)", 0.5, 5.0, 1.5, 0.1, disabled=playing
    )
    duration = day_delay if mode == "Day" else night_delay
    consent = st.checkbox(
        "I consent to saving one face screenshot per episode", value=False
    )
    emergency_config = settings.get("emergency", {})
    configured = bool(
        emergency_config.get("bot_token")
        and emergency_config.get("chat_id")
        and emergency_config.get("provider", "telegram") == "telegram"
    )
    messaging = st.checkbox(
        "Enable TEST alerts to the configured Telegram test recipient",
        value=False,
        disabled=not configured,
    )
    if not configured:
        st.caption(
            "TEST messaging is unavailable until emergency secrets contain a bot token and test chat_id."
        )
    st.caption(
        "An episode ends after 0.5 seconds of observed open eyes. No face is unknown, never closed."
    )
    ear_threshold = st.slider(
        "EAR threshold",
        0.1,
        0.4,
        0.3,
        0.01,
        disabled=playing or method != "EAR threshold",
    )
    st.caption(
        "The largest visible face is monitored. Both eyes must be classified closed."
    )
    if trained:
        st.success("Trained model available")
    else:
        st.info("Run Drowsiness_Model_Training.ipynb to create the trained model.")
    st.caption(
        "Experimental eye-closure indicator. It does not measure all causes of fatigue."
    )

proposed_configuration = (driver, method, duration, ear_threshold)
if playing and "active_configuration" in st.session_state:
    driver, method, duration, ear_threshold = st.session_state.active_configuration
else:
    st.session_state.active_configuration = proposed_configuration

if method == "Trained eye classifier" and not trained:
    st.warning(
        "No trained model yet. Run the notebook, or select EAR threshold in the sidebar."
    )
    st.stop()

try:
    model_file = str(MODEL_PATH) if method == "Trained eye classifier" else None
    model_stamp = MODEL_PATH.stat().st_mtime_ns if model_file else 0
    detector = load_detector(model_file, model_stamp)
except (OSError, ValueError, RuntimeError, KeyError) as exc:
    st.error(f"Could not load the detector: {exc}")
    st.stop()

owner = st.session_state.monitor_owner + ":" + driver.strip()
monitor_key = f"{owner}:{method}:{duration}:{ear_threshold}:{model_stamp}"
if st.session_state.get("monitor_key") != monitor_key:
    st.session_state.monitor_key = monitor_key
    delivery = (
        dispatcher_for(emergency_config["bot_token"], emergency_config["chat_id"])
        if configured
        else None
    )
    st.session_state.monitor = Monitor(
        store, owner, driver.strip() or "Driver", duration, delivery
    )
monitor = st.session_state.monitor
monitor.configure(consent, messaging, st.session_state.get("latest_location"))

live_tab, image_tab, video_tab, report_tab, shots_tab, results_tab = st.tabs(
    [
        "Live webcam",
        "Image / snapshot",
        "Video analysis",
        "Reports",
        "Consented screenshots",
        "Model insights",
    ]
)

with live_tab:
    st.subheader("Live browser camera")
    st.caption(
        "Enable the browser alarm below, then START the camera. Location sharing is optional; missing location never blocks a TEST alert."
    )
    st.caption(
        "HTTPS or localhost is required. Camera and location permissions are controlled by your browser."
    )
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.lock = Lock()
                self.latest = None
                self.monitor = monitor
                self.detector = detector
                self.threshold = ear_threshold

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                if image.shape[1] > 800:
                    image = cv2.resize(
                        image, (800, round(image.shape[0] * 800 / image.shape[1]))
                    )
                observation = self.detector.observe(image, self.threshold)
                now = time.monotonic()
                state = self.monitor.observe(
                    observation["closed"] if observation else None,
                    image,
                    timestamp=now,
                    box=observation["box"] if observation else None,
                )
                status = (
                    "NO FACE"
                    if observation is None
                    else "DROWSY"
                    if state.alarm
                    else "EYES CLOSED"
                    if observation["closed"]
                    else "EYES OPEN"
                )
                output = annotate(image, observation, status, state.closed_seconds)
                with self.lock:
                    self.latest = {
                        "time": now,
                        "status": status,
                        "ear": observation["ear"] if observation else None,
                        "state": state,
                    }
                return av.VideoFrame.from_ndarray(output, format="bgr24")

        stream_key = hashlib.sha256(monitor_key.encode()).hexdigest()[:12]
        rtc_config = settings.get(
            "rtc_configuration",
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        context = webrtc_streamer(
            key=f"eyes-{stream_key}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            rtc_configuration=rtc_config,
            async_processing=False,
        )

        @st.fragment(run_every=0.1)
        def live_status():
            changed = (
                st.session_state.get("stream_playing", False) != context.state.playing
            )
            st.session_state.stream_playing = context.state.playing
            if changed:
                st.rerun()
            row = None
            if context.video_processor:
                with context.video_processor.lock:
                    row = context.video_processor.latest
            fresh = bool(
                context.state.playing and row and time.monotonic() - row["time"] <= 0.5
            )
            if not fresh:
                monitor.observe(None)
            alarm = bool(fresh and row["state"].alarm)
            browser = (
                browser_controls(alarm=alarm, key="driver-browser", default={}) or {}
            )
            st.session_state.latest_location = browser.get("location")
            monitor.configure(consent, messaging, browser.get("location"))
            if browser.get("timezone"):
                try:
                    ZoneInfo(browser["timezone"])
                    st.session_state.browser_timezone = browser["timezone"]
                except (ValueError, KeyError):
                    pass
            if not browser.get("armed"):
                st.info(
                    "Browser alarm is disabled. Click Enable browser alarm to permit sound."
                )
            if fresh:
                cols = st.columns(3)
                cols[0].metric("State", row["status"])
                cols[1].metric(
                    "Continuous closure", f"{row['state'].closed_seconds:.1f}s"
                )
                cols[2].metric(
                    "EAR", f"{row['ear']:.3f}" if row["ear"] is not None else "—"
                )
                if row["status"] == "NO FACE":
                    st.warning(
                        "No face detected. Alarm is off and continuous closure has reset."
                    )
                elif alarm:
                    st.error("Drowsiness episode active: eyes closed.")
            elif context.state.playing:
                st.warning("Waiting for a fresh face observation; alarm off.")
            jobs = store.jobs(owner)
            if jobs:
                st.write("TEST delivery status")
                st.dataframe(jobs, hide_index=True)
            elif not messaging:
                st.caption("TEST messaging is disabled. No messages will be sent.")

        live_status()
    except ImportError:
        st.info(
            "Install requirements.txt for browser webcam streaming. Upload analysis remains available."
        )

with image_tab:
    st.subheader("Check a face image")
    source = st.radio("Image source", ["Upload", "Take a photo"], horizontal=True)
    picture = (
        st.file_uploader("Face photo", type=["jpg", "jpeg", "png"])
        if source == "Upload"
        else st.camera_input("Take a photo")
    )
    if picture:
        frame = cv2.imdecode(
            np.frombuffer(picture.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            st.error("Cannot decode this image. Try another JPG or PNG.")
        else:
            output, row = analyze_frame(
                detector, ClosureTracker(duration), frame, 0.0, ear_threshold
            )
            st.image(
                output,
                channels="BGR",
                caption=row["status"],
                width=min(800, max(320, output.shape[1])),
            )
            st.caption(
                "A photo shows eye state only; video is needed to measure sustained closure."
            )
            if row["faces"] > 1:
                st.info("Multiple faces detected; showing the largest face.")
            encoded = cv2.imencode(".png", output)[1].tobytes()
            st.download_button(
                "Download annotated image", encoded, "eye_result.png", "image/png"
            )

with video_tab:
    st.subheader("Analyze a recorded video")
    upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    st.caption(
        "Processes at most 120 seconds. Recorded videos use media time and never send live TEST messages or automatically save face screenshots."
    )
    settings = (method, duration, ear_threshold, model_stamp)
    identity = (
        (hashlib.sha256(upload.getvalue()).hexdigest(), settings) if upload else None
    )
    if st.button("Analyze video", disabled=upload is None):
        rows, last_frame = [], None
        tracker = EpisodeTracker(delay=duration)
        video_episodes = []
        preview, progress = st.empty(), st.progress(0.0)
        with tempfile.TemporaryDirectory(prefix="drowsiness-") as folder:
            path = Path(folder) / f"input{Path(upload.name).suffix}"
            path.write_bytes(upload.getvalue())
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    st.error(
                        "Cannot open this video. Try an MP4 with a supported codec."
                    )
                else:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0:
                        st.error(
                            "The video has no valid frame rate; closure durations cannot be measured."
                        )
                    else:
                        limit = max(1, int(fps * 120))
                        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                        expected = min(total, limit) if total > 0 else limit
                        for index in range(limit):
                            ok, frame = capture.read()
                            if not ok:
                                break
                            # Media timestamps provide a clock independent of inference speed.
                            if frame.shape[1] > 800:
                                frame = cv2.resize(
                                    frame,
                                    (800, round(frame.shape[0] * 800 / frame.shape[1])),
                                )
                            observation = detector.observe(frame, ear_threshold)
                            from datetime import timezone, timedelta

                            state = tracker.update(
                                observation["closed"] if observation else None,
                                index / fps,
                                datetime(2000, 1, 1, tzinfo=timezone.utc)
                                + timedelta(seconds=index / fps),
                            )
                            if state.started:
                                video_episodes.append(state.started)
                            status = (
                                "NO FACE"
                                if observation is None
                                else "DROWSY"
                                if state.alarm
                                else "EYES CLOSED"
                                if observation["closed"]
                                else "EYES OPEN"
                            )
                            last_frame = annotate(
                                frame, observation, status, state.closed_seconds
                            )
                            row = {
                                "time_seconds": index / fps,
                                "status": status,
                                "ear": observation["ear"] if observation else None,
                                "closed_seconds": state.closed_seconds,
                                "faces": observation["face_count"]
                                if observation
                                else 0,
                            }
                            rows.append(row)
                            if index % max(1, int(fps / 4)) == 0:
                                preview.image(
                                    last_frame, channels="BGR", width="stretch"
                                )
                                progress.progress(min((index + 1) / expected, 1.0))
            finally:
                capture.release()
        progress.empty()
        if rows:
            preview.empty()
            st.session_state.video_result = (identity, rows, last_frame)
        else:
            st.warning("No frames were analyzed.")
    saved = st.session_state.get("video_result")
    if saved and saved[0] == identity:
        _, rows, last_frame = saved
        cols = st.columns(3)
        cols[0].metric("Analyzed frames", len(rows))
        cols[1].metric("Frames with a face", sum(r["faces"] > 0 for r in rows))
        cols[2].metric("Drowsy frames", sum(r["status"] == "DROWSY" for r in rows))
        st.image(last_frame, channels="BGR", width="stretch")
        st.line_chart({"closed_seconds": [r["closed_seconds"] for r in rows]})
        st.caption("Horizontal axis: decoded frame index.")
        st.download_button(
            "Download frame results (CSV)",
            csv_bytes(rows),
            "video_results.csv",
            "text/csv",
        )

with report_tab:
    zone = st.session_state.get("browser_timezone", "Asia/Kolkata")
    selected_date = st.date_input(
        "Report date", value=datetime.now(ZoneInfo(zone)).date()
    )
    slots = report_rows(store.episodes(owner), selected_date, zone)
    st.write("Date:", selected_date.isoformat())
    st.write("Total episodes:", len(slots))
    st.dataframe(slots, column_order=["Start", "End"], hide_index=True)
    st.download_button(
        "Download PDF",
        report_pdf(selected_date, slots),
        f"episodes_{selected_date}.pdf",
        "application/pdf",
    )
    st.download_button(
        "Download episode CSV",
        report_csv(selected_date, slots),
        f"episodes_{selected_date}.csv",
        "text/csv",
    )

with shots_tab:
    st.caption(
        "Only screenshots saved with consent in this browser session and driver profile appear here. Deleting an image does not delete its episode count."
    )
    if st.button("Delete all my episode screenshots"):
        store.delete_all_screenshots(owner)
    for shot in store.screenshots(owner):
        st.image(shot["screenshot"], caption=shot["started_at"], width=320)
        if st.button("Delete this screenshot", key="delete-" + shot["id"]):
            store.delete_screenshot(owner, shot["id"])
            st.rerun()

with results_tab:
    report_path = ROOT / "reports" / "metrics.json"
    if not report_path.exists():
        report_path = ROOT / "experiments" / "deployment_metrics.json"
    if report_path.exists():
        try:
            metrics = json.loads(report_path.read_text())
            if (
                trained
                and metrics.get("model_sha256")
                != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            ):
                st.warning(
                    "The report belongs to a different model. Rerun the notebook to refresh results."
                )
                st.stop()
            st.subheader("Held-out eye-state classification results")
            cols = st.columns(3)
            for col, key in zip(cols, ("accuracy", "balanced_accuracy", "closed_f1")):
                col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")
            st.caption(
                "Scores measure eye-state classification, not clinical drowsiness detection."
            )
            st.json(metrics)
            for name in ("confusion_matrix.png", "example_predictions.png"):
                path = ROOT / "reports" / name
                if path.exists():
                    st.image(str(path))
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Cannot read training results: {exc}. Rerun the notebook.")
    else:
        st.info("No evaluation report yet. Run all notebook cells with your dataset.")
