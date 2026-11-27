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
    observation_status,
)
from episode_store import EpisodeStore
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
def load_detector(model_path, model_modified, inference_mode):
    # Timestamp invalidates the cached classifier after notebook training.
    return FaceEyeDetector(model_path=model_path, mode=inference_mode)


def csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


st.title("Drowsiness Monitor")
st.caption("Monitor eye closure, review recordings, and download episode reports.")
st.markdown(
    """<style>
.block-container {padding-top: 2.5rem; padding-bottom: 3rem;}
.stButton button, .stDownloadButton button {min-height: 2.75rem;}
[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
</style>""",
    unsafe_allow_html=True,
)
page = st.radio(
    "Workspace",
    [
        "Live webcam",
        "Image / snapshot",
        "Video analysis",
        "Reports",
        "Consented screenshots",
        "Model insights",
    ],
    horizontal=True,
    key="workspace",
)
if st.session_state.get("last_page", page) != page:
    st.session_state.stream_playing = False
    st.session_state.pop("https_result", None)
    st.session_state.pop("https_ack", None)
    if "monitor" in st.session_state:
        st.session_state.monitor.observe(None)
    playing = False
st.session_state.last_page = page
with st.sidebar:
    st.header("Detection settings")
    trained = MODEL_PATH.is_file()
    method = st.selectbox(
        "Detection method",
        ["Hybrid (recommended)", "EAR threshold", "SVM only (experimental)"],
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
        and not str(emergency_config.get("bot_token", "")).startswith("YOUR_")
        and not str(emergency_config.get("chat_id", "")).startswith("YOUR_")
    )
    messaging = st.checkbox(
        "Enable TEST alerts to the configured Telegram test recipient",
        help="Separate from screenshot consent. Requires a Telegram bot token and authorized test chat ID in Streamlit secrets.",
        value=False,
        disabled=not configured,
    )
    if not configured:
        st.info(
            "Telegram is not configured. Screenshot consent does not enable messaging."
        )
        with st.expander("How to enable Telegram TEST alerts"):
            st.write(
                "Add a bot token and your authorized test chat ID to Streamlit Secrets (or .streamlit/secrets.toml locally), then restart the app."
            )
            st.code(
                '[emergency]\nprovider = "telegram"\nbot_token = "YOUR_BOT_TOKEN"\nchat_id = "YOUR_TEST_CHAT_ID"',
                language="toml",
            )
    else:
        st.caption("Telegram configured. TEST alerts require this separate opt-in.")
    st.caption(
        "An episode ends after 0.5 seconds of observed open eyes. No face is unknown, never closed."
    )
    ear_threshold = st.slider(
        "EAR threshold",
        0.1,
        0.4,
        0.25,
        0.01,
        disabled=playing,
    )
    st.caption(
        "Hybrid uses clear eye geometry and checks SVM agreement near the threshold. Disagreement is uncertain and does not trigger an alert."
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

if method != "EAR threshold" and not trained:
    st.warning(
        "No trained model yet. Run the notebook, or select EAR threshold in the sidebar."
    )
    st.stop()

try:
    model_file = str(MODEL_PATH) if method != "EAR threshold" else None
    model_stamp = MODEL_PATH.stat().st_mtime_ns if model_file else 0
    detector = load_detector(
        model_file, model_stamp, "svm" if method.startswith("SVM") else "hybrid"
    )
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

if page == "Live webcam":
    st.subheader("Live browser camera")
    st.caption(
        "Enable the browser alarm below, then START the camera. Location sharing is optional; missing location never blocks a TEST alert."
    )
    st.caption(
        "HTTPS or localhost is required. Camera and location permissions are controlled by your browser."
    )
    transport = st.selectbox(
        "Camera connection",
        ["HTTPS camera (recommended)", "WebRTC (requires TURN on some networks)"],
        disabled=playing,
    )
    if transport.startswith("HTTPS"):
        st.caption(
            "Frames are analyzed over the app connection. Only consented episode screenshots are saved."
        )

        @st.fragment(run_every=0.2)
        def https_camera():
            previous = st.session_state.get("https_result", {})
            fresh = time.monotonic() - previous.get("received", 0) < 0.7
            value = (
                browser_controls(
                    camera=True,
                    ack=st.session_state.get("https_ack", 0),
                    camera_status=previous.get("status", "Waiting for camera"),
                    alarm=bool(fresh and previous.get("alarm")),
                    key="https-camera-controls",
                    default={},
                )
                or {}
            )
            active = bool(value.get("camera_active"))
            monitor.configure(consent, messaging, value.get("location"))
            st.session_state.latest_location = value.get("location")
            if value.get("timezone"):
                try:
                    ZoneInfo(value["timezone"])
                    st.session_state.browser_timezone = value["timezone"]
                except (ValueError, KeyError):
                    pass
            frame_id = value.get("frame_id")
            if frame_id and frame_id != st.session_state.get("https_ack"):
                st.session_state.https_ack = frame_id
                try:
                    from camera_transport import decode_camera_frame

                    image = decode_camera_frame(value)
                    observation = detector.observe(image, ear_threshold, upsample=0)
                    state = monitor.observe(
                        observation["closed"] if observation else None,
                        image,
                        box=observation["box"] if observation else None,
                    )
                    status = observation_status(observation, state.alarm)
                    st.session_state.https_result = {
                        "received": time.monotonic(),
                        "alarm": state.alarm,
                        "status": status,
                        "ear": observation["ear"] if observation else None,
                        "seconds": state.closed_seconds,
                        "reason": observation.get("reason", "") if observation else "",
                    }
                except (ValueError, cv2.error):
                    monitor.observe(None)
                    st.session_state.https_result = {
                        "status": "Frame delayed or invalid; waiting for a fresh frame",
                        "alarm": False,
                    }
            elif not fresh or not active:
                monitor.observe(None)
                st.session_state.https_result = {
                    "status": (
                        "Camera stopped" if not active else "Waiting for a fresh frame"
                    ),
                    "alarm": False,
                }
            result = st.session_state.get("https_result", {})
            status_text = result.get("status", "Waiting for camera")
            if status_text == "DROWSY":
                st.error("Drowsiness episode active — eyes closed")
            elif status_text in ("EYES CLOSED", "UNCERTAIN"):
                st.warning(status_text + ": " + result.get("reason", ""))
            elif status_text == "EYES OPEN":
                st.success("Eyes open")
            else:
                st.write(status_text)
            if result.get("ear") is not None:
                st.caption(
                    f"EAR: {result['ear']:.3f} · Continuous closure: {result['seconds']:.1f} s"
                )
            if result.get("status") == "NO FACE":
                st.warning(
                    "No face detected. Face the camera in good lighting; alarm is off."
                )
            if not value.get("armed"):
                st.info("Click Enable browser alarm to permit sound.")
            jobs = store.jobs(owner)
            if jobs:
                st.write("TEST delivery status")
                st.dataframe(jobs, hide_index=True)
            if st.session_state.get("stream_playing", False) != active:
                st.session_state.stream_playing = active
                st.rerun()

        https_camera()
    else:
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
                    observation = self.detector.observe(
                        image, self.threshold, upsample=0
                    )
                    now = time.monotonic()
                    state = self.monitor.observe(
                        observation["closed"] if observation else None,
                        image,
                        timestamp=now,
                        box=observation["box"] if observation else None,
                    )
                    status = observation_status(observation, state.alarm)
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
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 8, "max": 10},
                    },
                    "audio": False,
                },
                rtc_configuration=rtc_config,
                async_processing=False,
            )

            @st.fragment(run_every=0.1)
            def live_status():
                changed = (
                    st.session_state.get("stream_playing", False)
                    != context.state.playing
                )
                st.session_state.stream_playing = context.state.playing
                if changed:
                    st.rerun()
                row = None
                if context.video_processor:
                    with context.video_processor.lock:
                        row = context.video_processor.latest
                fresh = bool(
                    context.state.playing
                    and row
                    and time.monotonic() - row["time"] <= 0.5
                )
                if not fresh:
                    monitor.observe(None)
                alarm = bool(fresh and row["state"].alarm)
                browser = (
                    browser_controls(alarm=alarm, key="driver-browser", default={})
                    or {}
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
                    elif row["status"] == "UNCERTAIN":
                        st.warning(
                            "Eye state uncertain. Adjust lighting or sensitivity; alarm off."
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
if page == "Image / snapshot":
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
            st.info(row["status"] + " — " + row.get("reason", ""))
            st.caption(
                "A photo shows eye state only; video is needed to measure sustained closure."
            )
            if method.startswith("SVM"):
                st.warning(
                    "Experimental SVM: MRL eye crops differ from full-face photos. Use Hybrid for monitoring."
                )
            if row["faces"] > 1:
                st.info("Multiple faces detected; showing the largest face.")
            encoded = cv2.imencode(".png", output)[1].tobytes()
            st.download_button(
                "Download annotated image", encoded, "eye_result.png", "image/png"
            )

if page == "Video analysis":
    st.subheader("Analyze a recorded video")
    st.info(
        "Video review samples about 5 frames per second for faster processing. Very brief blinks between samples may be missed."
    )
    upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    st.caption(
        "Processes at most 120 seconds. Recorded videos use media time and never send live TEST messages or automatically save face screenshots."
    )
    video_settings = (method, duration, ear_threshold, model_stamp)
    identity = (
        (hashlib.sha256(upload.getvalue()).hexdigest(), video_settings)
        if upload
        else None
    )
    if st.button("Analyze video", disabled=upload is None, type="primary"):
        from video_analysis import analyze_video

        preview, progress = st.empty(), st.progress(0.0)
        last_preview = [0.0]

        def update_video(fraction, frame):
            progress.progress(fraction)
            if time.monotonic() - last_preview[0] > 0.5:
                preview.image(frame, channels="BGR", width="stretch")
                last_preview[0] = time.monotonic()

        try:
            with tempfile.TemporaryDirectory(prefix="drowsiness-") as folder:
                path = Path(folder) / f"input{Path(upload.name).suffix}"
                path.write_bytes(upload.getvalue())
                with st.spinner("Analyzing sampled frames using video time…"):
                    result = analyze_video(
                        path, detector, duration, ear_threshold, update_video
                    )
                st.session_state.video_result = (identity, result)
        except (ValueError, cv2.error, OSError) as exc:
            st.session_state.pop("video_result", None)
            st.error(str(exc))
        finally:
            progress.empty()
            preview.empty()
    saved = st.session_state.get("video_result")
    if saved and saved[0] == identity:
        result = saved[1]
        rows = result["rows"]
        cols = st.columns(3)
        cols[0].metric("Analyzed time", f"{result['analyzed_seconds']:.1f} s")
        cols[1].metric("Sampled frames", len(rows))
        cols[2].metric("Episodes", len(result["episodes"]))
        st.caption(
            f"Sample rate: {result['sample_fps']:.1f} fps. Durations use the video timeline. Clips longer than 120 s are limited to the first 120 s."
        )
        st.image(result["last_frame"], channels="BGR", width="stretch")
        if result["episodes"]:
            st.dataframe(result["episodes"], hide_index=True)
            st.download_button(
                "Download video episodes (CSV)",
                csv_bytes(result["episodes"]),
                "video_episodes.csv",
                "text/csv",
            )
        else:
            st.info(
                "No sustained eye-closure episodes detected in the analyzed interval."
            )
        import pandas as pd

        st.line_chart(
            pd.DataFrame(rows).set_index("time_seconds")[["closed_seconds"]],
            x_label="Video time (seconds)",
        )
        st.download_button(
            "Download frame results (CSV)",
            csv_bytes(rows),
            "video_results.csv",
            "text/csv",
        )

if page == "Reports":
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

if page == "Consented screenshots":
    st.caption(
        "Only screenshots saved with consent in this browser session and driver profile appear here. Deleting an image does not delete its episode count."
    )
    if st.button("Delete all my episode screenshots"):
        store.delete_all_screenshots(owner)
        st.rerun()
    for shot in store.screenshots(owner):
        st.image(shot["screenshot"], caption=shot["started_at"], width=320)
        if st.button("Delete this screenshot", key="delete-" + shot["id"]):
            store.delete_screenshot(owner, shot["id"])
            st.rerun()

if page == "Model insights":
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
                "These are the saved SVM-only results on MRL eye crops. Hybrid full-face inference has not been benchmarked; these scores do not measure video drowsiness accuracy."
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
