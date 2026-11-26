"""Run with: streamlit run streamlit_app.py"""

import csv
import hashlib
import io
import json
import tempfile
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import streamlit as st

from detection import MODEL_PATH, ROOT, ClosureTracker, FaceEyeDetector, analyze_frame

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
    method = st.selectbox("Detection method", ["Trained eye classifier", "EAR threshold"],
                          index=0 if trained else 1)
    duration = st.slider("Eye closure before alert (seconds)", 0.5, 4.0, 1.5, 0.1)
    ear_threshold = st.slider("EAR threshold", 0.1, 0.4, 0.3, 0.01,
                              disabled=method != "EAR threshold")
    st.caption("The largest visible face is monitored. Both eyes must be classified closed.")
    if trained:
        st.success("Trained model available")
    else:
        st.info("Run Drowsiness_Model_Training.ipynb to create the trained model.")
    st.caption("Experimental eye-closure indicator. It does not measure all causes of fatigue.")

if method == "Trained eye classifier" and not trained:
    st.warning("No trained model yet. Run the notebook, or select EAR threshold in the sidebar.")
    st.stop()

try:
    model_file = str(MODEL_PATH) if method == "Trained eye classifier" else None
    model_stamp = MODEL_PATH.stat().st_mtime_ns if model_file else 0
    detector = load_detector(model_file, model_stamp)
except (OSError, ValueError, RuntimeError, KeyError) as exc:
    st.error(f"Could not load the detector: {exc}")
    st.stop()

live_tab, image_tab, video_tab, results_tab = st.tabs(
    ["Live webcam", "Image / snapshot", "Video analysis", "Training results"])

with live_tab:
    st.subheader("Live browser camera")
    st.write("Press START and allow camera access. A red border marks sustained eye closure.")
    st.caption("Use localhost or HTTPS. Remote hosting may require STUN/TURN configuration.")
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.tracker = ClosureTracker(duration=duration)
                self.lock = Lock()
                self.latest = None

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                output, row = analyze_frame(detector, self.tracker, image,
                                            time.monotonic(), ear_threshold)
                with self.lock:
                    self.latest = row
                return av.VideoFrame.from_ndarray(output, format="bgr24")

        stream_key = hashlib.sha256(
            f"{method}:{duration}:{ear_threshold}:{model_stamp}".encode()
        ).hexdigest()[:12]
        # Empty ICE servers work locally. Configure remote ICE servers in secrets.toml.
        rtc_config = st.secrets.get("rtc_configuration", {"iceServers": []}) if (
            ROOT / ".streamlit" / "secrets.toml").exists() else {"iceServers": []}
        context = webrtc_streamer(
            key=f"eyes-{stream_key}", mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": {"width": {"ideal": 640},
                                                 "height": {"ideal": 480}}, "audio": False},
            rtc_configuration=rtc_config, async_processing=True)
        sound_enabled = st.checkbox("Play alert sound", value=False)

        @st.fragment(run_every=0.5)
        def live_status():
            if not context.state.playing or not context.video_processor:
                return
            with context.video_processor.lock:
                row = context.video_processor.latest
            if row and time.monotonic() - row["time_seconds"] <= 2.0:
                cols = st.columns(3)
                cols[0].metric("State", row["status"])
                cols[1].metric("Eye closure", f"{row['closed_seconds']:.1f}s")
                cols[2].metric("EAR", f"{row['ear']:.3f}" if row["ear"] is not None else "—")
                if row["status"] == "DROWSY":
                    st.error("Sustained eye closure detected.")
                    if sound_enabled and (ROOT / "audio/alert.wav").exists():
                        st.audio(str(ROOT / "audio/alert.wav"), autoplay=True, loop=True)

        live_status()
    except ImportError:
        st.info("Install requirements.txt for live video. Image and uploaded-video analysis still work.")

with image_tab:
    st.subheader("Check a face image")
    source = st.radio("Image source", ["Upload", "Take a photo"], horizontal=True)
    picture = (st.file_uploader("Face photo", type=["jpg", "jpeg", "png"])
               if source == "Upload" else st.camera_input("Take a photo"))
    if picture:
        frame = cv2.imdecode(np.frombuffer(picture.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Cannot decode this image. Try another JPG or PNG.")
        else:
            output, row = analyze_frame(detector, ClosureTracker(duration), frame, 0.0, ear_threshold)
            st.image(output, channels="BGR", caption=row["status"],
                     width=min(800, max(320, output.shape[1])))
            st.caption("A photo shows eye state only; video is needed to measure sustained closure.")
            if row["faces"] > 1:
                st.info("Multiple faces detected; showing the largest face.")
            encoded = cv2.imencode(".png", output)[1].tobytes()
            st.download_button("Download annotated image", encoded, "eye_result.png", "image/png")

with video_tab:
    st.subheader("Analyze a recorded video")
    upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    st.caption("Processes at most 120 seconds per run. Results contain one row per decoded frame.")
    settings = (method, duration, ear_threshold, model_stamp)
    identity = (hashlib.sha256(upload.getvalue()).hexdigest(), settings) if upload else None
    if st.button("Analyze video", disabled=upload is None):
        rows, last_frame = [], None
        tracker = ClosureTracker(duration=duration)
        preview, progress = st.empty(), st.progress(0.0)
        with tempfile.TemporaryDirectory(prefix="drowsiness-") as folder:
            path = Path(folder) / f"input{Path(upload.name).suffix}"
            path.write_bytes(upload.getvalue())
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    st.error("Cannot open this video. Try an MP4 with a supported codec.")
                else:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not np.isfinite(fps) or fps <= 0:
                        st.error("The video has no valid frame rate; closure durations cannot be measured.")
                    else:
                        limit = max(1, int(fps * 120))
                        total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                        expected = min(total, limit) if total > 0 else limit
                        for index in range(limit):
                            ok, frame = capture.read()
                            if not ok:
                                break
                            last_frame, row = analyze_frame(detector, tracker, frame, index / fps, ear_threshold)
                            rows.append(row)
                            if index % max(1, int(fps / 4)) == 0:
                                preview.image(last_frame, channels="BGR", width="stretch")
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
        st.download_button("Download frame results (CSV)", csv_bytes(rows), "video_results.csv", "text/csv")

with results_tab:
    report_path = ROOT / "reports" / "metrics.json"
    if report_path.exists():
        try:
            metrics = json.loads(report_path.read_text())
            if trained and metrics.get("model_sha256") != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
                st.warning("The report belongs to a different model. Rerun the notebook to refresh results.")
                st.stop()
            st.subheader("Held-out eye-state classification results")
            cols = st.columns(3)
            for col, key in zip(cols, ("accuracy", "balanced_accuracy", "closed_f1")):
                col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")
            st.caption("Scores measure eye-state classification, not clinical drowsiness detection.")
            st.json(metrics)
            for name in ("confusion_matrix.png", "example_predictions.png"):
                path = ROOT / "reports" / name
                if path.exists():
                    st.image(str(path))
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"Cannot read training results: {exc}. Rerun the notebook.")
    else:
        st.info("No evaluation report yet. Run all notebook cells with your dataset.")
