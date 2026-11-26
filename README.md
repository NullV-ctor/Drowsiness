# Driver Drowsiness Detection

Train an open/closed-eye classifier in one Jupyter notebook and use a Streamlit frontend for browser webcam, photo, and recorded-video analysis. The original OpenCV webcam scripts are also available.

The existing `shape_predictor_68_face_landmarks.dat` is a **pretrained face landmark predictor**, not a training dataset. The repository contains demo images and screenshots, but no labeled training set. The app can run immediately using the existing eye aspect ratio (EAR) method. Training the new eye classifier requires labeled single-eye images.

## Setup

Python 3.11 or 3.12 is recommended. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux, building dlib may require `cmake`, a C++ compiler, and Python development headers. Install only one OpenCV distribution in the environment; the requirements use `opencv-python`.

The extracted `shape_predictor_68_face_landmarks.dat` is already present in this checkout. If missing, extract the supplied archive:

```bash
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```

The original predictor is available from [dlib](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Check the source model and dataset terms for your intended use.

## Train and output results

Put **single-eye crops**, labeled by actual eye state, into:

```text
data/eyes/
  open/
    eye_001.jpg
    ...
  closed/
    eye_002.jpg
    ...
```

`Open` and `Closed` folder names are accepted too. For subject/session-independent evaluation, organize both class folders by the same identifiers, such as `open/person_01/` and `closed/person_01/`. All images from a person/session stay in one partition; use at least five independent groups. Flat folders use an image-level split, which may overestimate performance when identities or adjacent video frames overlap.

Existing splits are supported as `train/`, `validation/` (or `val/`), and `test/`, each containing `open/` and `closed/`. Do not use full-face Alert/Drowsy images as eye crops. For your own captures, use the padded eye-cropping convention in `detection.crop_eye`.

```bash
jupyter lab Drowsiness_Model_Training.ipynb
```

Select a Python 3.11 or 3.12 kernel and run the notebook's **0. Check the notebook environment** cell. It checks installed versions, restores the compatible NumPy/OpenCV pair, and installs missing training packages into the selected kernel. If old binary libraries are still loaded, it prints a restart-needed status and skips later stages safely. Restart the kernel, then Run All. There is no intentional error traceback. Set `DATA_DIR` in the configuration cell before training.

Run All also handles missing or invalid datasets with a clear **Training pending** message. Training, evaluation, and export are skipped until labeled data passes validation; no new model or accuracy metrics are generated in that case. The final cell summarizes what actually completed.

If you see missing `cv2`, `_ARRAY_API not found`, or `numpy.dtype size changed`, run the setup cell or this command in a notebook cell, then restart the kernel:

```python
%pip install "numpy==1.26.4" "opencv-python==4.10.0.84"
```

The `%pip` command targets the active Jupyter kernel. A terminal's `pip` may install into a different environment. You can check the notebook's interpreter with `import sys; print(sys.executable)`. Keep the two version pins together: installing a newer OpenCV alone can upgrade NumPy to 2.x and break older compiled Anaconda packages.

If your environment is missing from Jupyter's kernel list:

```bash
python -m ipykernel install --user --name drowsiness --display-name "Drowsiness"
```

The notebook includes data checks, duplicate detection, class examples, train/validation/test splitting, shared HOG preprocessing, full SVM training, validation-based tuning, and held-out evaluation. It outputs:

- `models/eye_state_model.npz`: trained classifier for the app.
- `reports/metrics.json`: accuracy, balanced accuracy, closed-eye F1, ROC AUC, and class report.
- `reports/confusion_matrix.png` and `reports/example_predictions.png`.
- `reports/test_predictions.csv`, `reports/validation_trials.csv`, and `reports/split_manifest.csv`.

No results are fabricated or prefilled. The dlib landmark predictor remains pretrained; the notebook trains the eye-state classifier. Eye-state accuracy does not establish drowsiness detection accuracy. Training files, generated model files, and reports are ignored by Git by default.

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The sidebar selects **Trained eye classifier** after training or **EAR threshold** before training, with adjustable closure duration and EAR threshold.

- **Live webcam:** browser camera with annotated face/eye contours, live status, closure timer, and optional browser alert sound. Press START/STOP to control the stream. Changing detection settings starts a fresh stream. Browser autoplay rules may require interaction before sound plays.
- **Image / snapshot:** upload a face image or take a browser photo; download its annotated result. A single photo shows eye state, not sustained drowsiness.
- **Video analysis:** process up to the first 120 seconds using the video frame rate for timestamps; inspect summary counts and download per-frame CSV results.
- **Training results:** view exported evaluation metrics and plots.

The largest face is monitored. In classifier mode both eyes must be predicted closed; in EAR mode the mean EAR must fall below the threshold. A continuous closed-eye interval triggers the drowsiness alert. Missing faces, long observation gaps, and large face-box changes reset the timer. Face boxes are a simple continuity check, not identity tracking.

Browser webcam requires localhost or HTTPS. For remote hosting, configure your deployment's STUN/TURN servers in `.streamlit/secrets.toml`:

```toml
[rtc_configuration]
iceServers = [{ urls = ["stun:your-stun-server:3478"] }]
```

Do not commit secrets. Some networks require authenticated TURN in addition to STUN; see the [streamlit-webrtc deployment guide](https://github.com/whitphx/streamlit-webrtc#network-connectivity).

## Original scripts

Run from the project folder in the installed environment:

```bash
python face_and_eye_detector_single_image.py
python face_and_eye_detector_webcam_video.py
python drowsiness_detect.py
```

The original webcam detector uses EAR, plays `audio/alert.wav`, and saves screenshots. The Streamlit app does not automatically save webcam images.

## Verification

```bash
python -m unittest discover -s tests -v
```

The checks cover closure timing, missing-face and gap resets, face changes, feature consistency, and exported model parity with scikit-learn. Real model quality must be measured on an independent labeled dataset. Glasses, lighting, pose, and crop differences can affect predictions. This is an experimental eye-closure project.

Original project: [mohitwildbeast/Driver-Drowsiness-Detector](https://github.com/mohitwildbeast/Driver-Drowsiness-Detector). EAR implementation inspired by Adrian Rosebrock's PyImageSearch tutorial.
