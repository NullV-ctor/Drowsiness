# Driver Drowsiness Detection

Train an open/closed-eye classifier in one Jupyter notebook and use a Streamlit frontend for browser webcam, photo, and recorded-video analysis. The original OpenCV webcam scripts are also available.

The existing `shape_predictor_68_face_landmarks.dat` is a **pretrained face landmark predictor**, not a training dataset. The source repository contains demo images and screenshots. Raw labeled data and generated models/reports are kept locally under Git-ignored folders; an MRL importer is provided. The app can run immediately using the existing eye aspect ratio (EAR) method. Training the new eye classifier requires labeled single-eye images.

## Project files and why they are here

Start with `streamlit_app.py` to use the application, or
`Drowsiness_Model_Training.ipynb` to prepare and train the eye-state classifier.
The original desktop demonstrations are retained as separate examples.

### Source code and project configuration

| File | Purpose and reason for keeping it |
| --- | --- |
| `README.md` | Explains setup, training, application usage, verification, and the purpose of the project files. |
| `Drowsiness_Model_Training.ipynb` | Holds the complete training workflow: environment checks, labeled-data validation, preprocessing, splitting, SVM tuning, evaluation, and model export. It now uses strict person-held-out testing, GroupKFold tuning, and training-only augmentation; actual results are generated only when data is present. |
| `streamlit_app.py` | Main frontend. Provides browser webcam, image/snapshot, recorded-video analysis, downloads, and training-result views. Run this file with Streamlit. |
| `detection.py` | Shared face/eye detection, eye cropping, EAR calculation, sustained-closure timing, and frame annotation used by Streamlit. Keeps detection logic separate from interface code. |
| `eye_model.py` | Shared eye preprocessing, HOG feature extraction, and loading/prediction for the exported classifier. Sharing these functions keeps training and application preprocessing consistent. |
| `drowsiness_detect.py` | Original desktop webcam detector. Uses EAR, plays an alarm, and automatically saves drowsiness screenshots. Useful when running directly through OpenCV instead of the browser. |
| `face_and_eye_detector_single_image.py` | Original Haar-cascade demonstration for locating faces and eyes in `images/test.jpeg`. Shows basic image detection without classifier training. |
| `face_and_eye_detector_webcam_video.py` | Original Haar-cascade webcam demonstration. Shows basic face and eye boxes in an OpenCV window. |
| `tests/test_pipeline.py` | Checks closure timing and resets, eye geometry/cropping, preprocessing consistency, and exported classifier prediction parity. Helps catch regressions. |
| `prepare_mrl.py` | Imports the official MRL ZIP into class/person folders, remaps labels explicitly, and records source metadata plus the archive hash. |
| `training.py` | Shared subject-aware data audit, EDA, splitting, augmentation, GroupKFold tuning, held-out evaluation, and NPZ export used by the notebook and CLI. |
| `compare_methods.py` | Optional matched full-face benchmark for SVM versus EAR; tunes EAR on validation people only and reports test detection coverage. |
| `experiments/mrl_eye_results.json` | Compact record of the actual MRL data audit, configuration, and held-out results; raw data stays outside Git. |
| `tests/test_training.py` | Tests MRL label mapping, safe archive import, subject isolation, conflict handling, fold-only augmentation, exports, and matched-method threshold separation. |
| `episodes.py` | Pure fake-clock-testable episode state machine with reopening hysteresis and continuous emergency timing. |
| `monitoring.py` | Connects observations to one-shot episode recording, consented capture, and background TEST alert enqueueing. |
| `episode_store.py` | Session-scoped SQLite episode records, deletable JPEG blobs, and an idempotent delivery outbox. |
| `emergency_alerts.py` | Validates fresh browser location, formats TEST messages, and retries Telegram delivery outside video processing. |
| `episode_reports.py` | Generates date/time-slot/count-only daily PDF and CSV outputs. |
| `browser_controls/index.html` | Browser Web Audio alarm, watchdog, explicit audio gesture, and permission-based geolocation. |
| `.streamlit/secrets.example.toml` | Placeholder-only example for the TEST recipient, bot token, and hosted ICE configuration. |
| `tests/test_episodes.py` | Fake-clock episode, consent, report, location, retry, and nonblocking-delivery tests. |
| `requirements-dev.txt` | Installs runtime dependencies plus the PDF test reader and formatter for development checks. |
| `tests/test_app.py` | Renders Streamlit with mocked browser components and secrets; checks opt-in controls, report rendering, and failed-message status. |
| `tests/test_browser_controls.js` | Detached JavaScript tests using mocked audio, location, and browser events. |
| `requirements.txt` | Lists runtime and notebook dependencies. NumPy/OpenCV pins prevent the compatibility problem encountered in Anaconda. |
| `pyproject.toml` | Records Black's Python formatting rules: 88-character target line length, Python 3.11 syntax, and exclusions for virtual environments and notebook backups. |
| `.editorconfig` | Gives supporting editors consistent encoding, line endings, indentation, and whitespace rules. |
| `.gitignore` | Excludes virtual environments, Python caches, notebook backups, local secrets, training data, and generated models/reports from ordinary Git tracking. |

### Models and detector assets

| File | Purpose and reason for keeping it |
| --- | --- |
| `shape_predictor_68_face_landmarks.dat` | Extracted pretrained dlib model used to locate facial landmarks, including the eyes. Required by the EAR/landmark detector. It is model weights, not labeled training data. |
| `shape_predictor_68_face_landmarks.dat.bz2` | Compressed copy of that model. Kept so the extracted `.dat` file can be restored. The running application reads the extracted file. |
| `haarcascades/haarcascade_frontalface_default.xml` | Pretrained OpenCV face detector used by the original demonstrations and the original desktop detector's face-box overlay. |
| `haarcascades/haarcascade_eye.xml` | Pretrained OpenCV eye detector used by the two original Haar-cascade demonstrations. |
| `audio/alert.wav` | Alert sound used by the original desktop detector and optional Streamlit browser audio. |

The pretrained XML files and binary assets retain their supplied contents. Formatting
applies to project source and notebook code, rather than rewriting detector data.

### Example images and existing results

These files are examples and visual references. They are not a labeled training dataset.

| File | Purpose and reason for keeping it |
| --- | --- |
| `images/test.jpeg` | Input photo used by the original single-image detector and application smoke checks. |
| `images/result_face_detector_single_image.png` | Stored face/eye detection result for illustrating the single-image demonstration. |
| `images/result_output.png` | Stored result image retained as a visual reference for the project. |
| `images/ear_analysis_result.png` | Stored EAR analysis illustration/result retained for reference. |
| `images/Image_screenshot_20.11.2026.png` | Saved image-demonstration screenshot retained for documentation. |
| `images/Video_screenshot_20.11.2026.png` | Saved video-demonstration screenshot retained for documentation. |

### Captured screenshots

The original `drowsiness_detect.py` writes `screenshots/drowsy_YYYYMMDD_HHMMSS.png`
when its alert is triggered, subject to the screenshot cooldown. Each timestamp gives
an event a separate filename. Streamlit does not automatically create these captures.

| Existing file | Purpose |
| --- | --- |
| `screenshots/Video_screenshot_20.11.2026.png` | Stored video-demonstration screenshot retained as a visual reference. |
| `screenshots/drowsy_20261121_005849.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_005856.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_005905.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_005913.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_005926.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_005959.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_010005.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_010015.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261121_010103.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145655.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145704.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145714.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145723.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145731.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145739.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_145748.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_175126.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |
| `screenshots/drowsy_20261126_175159.png` | Timestamped capture saved by the original drowsiness detector; retained for reviewing an alert event. |

### Generated and environment folders

| Folder or file pattern | Why it exists |
| --- | --- |
| `venv/` | Local Python environment containing its interpreter, installed third-party packages, activation scripts, and environment metadata. Its individual package files belong to those dependencies and are not project source. |
| `__pycache__/*.pyc` | Python-generated bytecode caches for imported modules, including interpreter-specific copies. These speed up imports and can be regenerated. |
| `tests/__pycache__/*.pyc` | Bytecode caches generated while importing/running tests. |
| `.ipynb_checkpoints/Drowsiness_Model_Training-checkpoint.ipynb` | Jupyter's backup checkpoint of the training notebook. Edit the main notebook; this backup is managed by Jupyter. |
| `.git/` | Repository history, branches, index, and Git metadata. Git manages its internal files. |
| `.agents/`, `.codex/`, `.aws/` | Managed environment/configuration directories supplied by the workspace. They are not part of the detector's training or prediction pipeline. |

### Files created after real training

These paths are outputs or inputs expected by the pipeline, rather than files supplied
as a labeled dataset in this checkout.

| Path | Purpose |
| --- | --- |
| `data/eyes/` | Location for the labeled Open/Closed eye images you supply. |
| `models/eye_state_model.npz` | Trained SVM coefficients and scaler parameters exported for application inference. |
| `reports/metrics.json` | Held-out test scores, class report, training settings, and exported-model fingerprint. |
| `reports/confusion_matrix.png` | Test confusion matrix and ROC plot. |
| `reports/example_predictions.png` | Example test images with true and predicted eye states. |
| `reports/test_predictions.csv` | Per-image predictions and decision margins for the test partition. |
| `reports/validation_trials.csv` | Validation results for the SVM regularization values tested. |
| `reports/split_manifest.csv` | Records which images belong to each data partition. |

### Formatting the source again

With Black installed in your development environment (`black[jupyter]` includes
notebook-formatting support), run from the project folder:

```bash
python -m black .
```

The Python style configuration excludes the virtual environment and notebook backups.
The notebook's code cells were formatted with the same rules; its Markdown, stored
outputs, and cell metadata were retained. Vendor assets and generated files are not
formatting targets. No files were relocated or removed.

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

The eye classifier uses HOG features and a linear SVM. It requires person identities:
there is no random image-level split fallback. The existing dlib landmark predictor
remains pretrained.

### Import MRL Eye

Review the [MRL Eye provider's terms](https://mrl.cs.vsb.cz/eyedataset.html), download
its original archive, and save it as `data/raw/mrlEyes_2018_01.zip`. Then run:

```bash
python prepare_mrl.py data/raw/mrlEyes_2018_01.zip
```

The importer produces `data/eyes/open/s0001/...` and `data/eyes/closed/s0001/...`,
plus `mrl_manifest.csv` and `provenance.json`. It maps MRL's **closed=0/open=1** to
this project's **open=0/closed=1**, preserves subject IDs and the original metadata,
and records the downloaded archive's SHA-256 hash. It does not infer a license.

For custom single-eye crops, use:

```text
data/eyes/
  open/
    person_01/eye_001.png
    person_02/eye_002.png
  closed/
    person_01/eye_003.png
    person_02/eye_004.png
```

Use the same person identifier across every session of that person. Predefined
`train/`, `validation/` (or `val/`), and `test/` folders are also supported, each
with `open/person_id/` and `closed/person_id/`. Their people must be disjoint.
Predefined train and validation images are pooled for development CV; test people
remain untouched. Flat MRL filenames are accepted because they encode subject IDs;
arbitrary flat images without identities are rejected.

### Run the experiment

```bash
jupyter lab Drowsiness_Model_Training.ipynb
```

Select a Python 3.11/3.12 kernel, run setup, and configure `DATA_DIR`, the development
sample cap, and CV folds. If dependencies change while old binary modules are loaded,
restart the kernel and Run All. The NumPy/OpenCV pins remain together to avoid the
previous ABI incompatibility. The notebook does not fabricate results when data is absent.

The same workflow is available without Jupyter:

```bash
python training.py --data data/eyes --output . --cap 100 --folds 5
```

The initial CPU experiment uses at most **100 development images per person/class**.
All retained images from held-out test people are evaluated. Use `--cap 0` (or notebook
cap `None`) for full development training; record this as a new experiment rather
than silently changing the reported protocol.

The workflow:

1. Audits every original image: readability, dimensions, contrast, raw brightness,
   exact duplicates, and class/subject metadata.
2. Excludes every copy of exact pixel images carrying conflicting labels, without
   guessing a replacement label. The default standalone scanner is strict; the
   experiment explicitly selects and records this exclusion policy. Ordinary exact
   duplicates are deduplicated. Low-contrast images are flagged and retained.
3. Reserves approximately 20% of people for testing before model selection. No person
   appears in both development and test, or both sides of a CV fold.
4. Extracts the shared 64×32 equalized HOG representation. Evaluates 5 C values,
   2 class-weight settings, and 2 augmentation settings in 5-fold GroupKFold.
   Horizontal flips and small rotations are applied only to each fold's training
   originals; validation/test samples are never augmented. Scaling is fitted inside
   each training fold.
5. Selects the configuration with the best mean validation balanced accuracy, with
   F1 as a tie-breaker. Only configurations that converge in every fold are eligible.
   Refits on development data and evaluates the held-out test once.
6. Exports a numeric-only NPZ compatible with the existing app and verifies prediction
   parity with scikit-learn before replacing the model.

### Generated evidence

| Output | Evidence |
| --- | --- |
| `models/eye_state_model.npz` | App-compatible trained model. |
| `reports/metrics.json` | Test metrics, class report, selected settings, software versions, subjects, sample cap, dataset fingerprint, and model hash. |
| `reports/eda_summary.json` | Class counts before/after cleaning, person count, brightness statistics, and quality-issue counts. |
| `reports/eda_overview.png`, `reports/sample_grid.png` | Class distribution, images per person, brightness histograms, and original samples. |
| `reports/image_inventory.csv`, `reports/data_quality.csv` | Audited image metadata and every flagged/excluded image. |
| `reports/images_per_person.csv`, `reports/split_manifest.csv` | Person/class counts and exact partition/sample selection. |
| `reports/cv_membership.csv`, `reports/cv_fold_results.csv` | Original-image fold roles, per-fold scores, iterations, and convergence flags. |
| `reports/validation_trials.csv` | Mean and standard deviation of GroupKFold validation scores for every configuration. |
| `reports/model_comparison.csv` | SVM and majority baseline on identical test images. |
| `reports/confusion_matrix.png`, `reports/example_predictions.png` | Test confusion matrix/ROC and sample predictions. |
| `reports/test_predictions.csv`, `reports/test_metrics_per_person.csv` | Test predictions and per-person performance. |
| `reports/test_metrics_by_*.csv` | MRL subgroup scores for glasses, lighting, sensor, and gender codes. |
| `reports/ear_comparison.json`, `reports/video_evaluation.json` | Explicit not-run reasons until appropriate additional benchmarks are supplied. |

Raw images, imported data, trained models, and detailed reports are ignored by Git.
The concise `experiments/mrl_eye_results.json` records the executed experiment for
review without committing the raw dataset.

### EAR and full-video evaluation

MRL contains infrared eye crops, whereas the current dlib EAR pipeline needs a full
face. It cannot honestly be scored on MRL as a comparable method with this pipeline.
The SVM/majority baseline comparison is available immediately; an EAR comparison
requires a common labeled full-face benchmark. Supply a CSV with:

```text
path,subject_id,split,label
faces/driver_a_open.png,driver_a,validation,0
faces/driver_a_closed.png,driver_a,validation,1
faces/driver_b_open.png,driver_b,test,0
faces/driver_b_closed.png,driver_b,test,1
```

Paths are relative to the CSV. Label `1` means both eyes closed. Use single-driver
frames and globally consistent person IDs; benchmark test people must not occur in
classifier development. Run in the dlib-enabled app environment:

```bash
python compare_methods.py data/face_benchmark.csv --model models/eye_state_model.npz
```

EAR threshold selection uses validation people only. Both methods are scored on
exactly the same detectable test frames, with missed-face counts and detection
coverage reported separately under `reports/matched_faces/`.

Eye accuracy is not drowsiness accuracy. Full-video testing needs independent
held-out drivers, media timestamps, and human-labeled closure intervals/drowsiness
or alert events. False alerts per minute, missed events, and alert latency must be
measured separately. No such ground truth has been supplied for this step.

MRL's infrared sensors, uneven per-person sampling, demographic imbalance, and
correlated neighboring frames limit generalization to RGB browser webcams. The
source page's demographic summary should be distinguished from filename metadata
counts measured in the downloaded archive; discrepancies should be reported, not
silently corrected. Review the actual EDA and subgroup outputs before drawing
conclusions about driver fatigue.

## Executed MRL experiment

This is a real CPU experiment with capped development data, not a full-development
training claim. Imported 84,898 images from 37 people. Quality checks retained
84,827 images after excluding 66 conflicting-label images and 5 ordinary duplicates;
321 low-contrast images were flagged and retained.

The model used **5,107 development originals from 29 people** (cap 100 per person/class)
and **all 24,142 retained test images from 8 different people**. Five-fold GroupKFold
compared 20 configurations; all 100 fits converged. Selected C=0.001, balanced class
weights, augmentation disabled. The best mean CV balanced accuracy was
0.789172; the selected configuration's fold variability is in
`reports/validation_trials.csv`. Augmentation was evaluated only on training folds;
it did not win this experiment. Test observations were not used to select settings.

| Held-out test metric | Measured value |
| --- | --- |
| Accuracy | 0.877144 |
| Balanced accuracy | 0.878368 |
| Closed-eye precision | 0.842155 |
| Closed-eye recall | 0.918845 |
| Closed-eye F1 | 0.878830 |
| ROC-AUC | 0.953306 |
| Majority baseline accuracy | 0.484881 |

See `experiments/mrl_eye_results.json` for the source/archive hash, audit, exact subjects,
software versions, settings, and metrics. Detailed plots and per-image/fold evidence
are in `reports/`. The numeric model is in `models/eye_state_model.npz` and is already
compatible with the existing Streamlit app.

These scores apply to held-out MRL infrared eye crops in this experiment. EAR comparison,
RGB browser-camera generalization, and video-level drowsiness accuracy have not been
measured. No labeled full-face or full-video benchmark was supplied. The importer
observed gender codes 0/1 for 32/5 people, while the provider's webpage describes
33/4; this metadata discrepancy is recorded rather than silently altered.

## Clock based episodes and browser alerts

An episode is created after continuously observed closed eyes reach the selected delay
(default **day 2.0 seconds**, **night 1.5 seconds**). Its recorded start is the first
closed-eye observation, rather than the later alarm trigger. It ends only after **0.5
seconds of continuously observed open eyes**. A shorter reopening keeps the same episode,
but stops audio immediately and resets the independent continuous-closure emergency timer.
Wall-clock timestamps are stored in UTC; the Reports tab uses the browser timezone when
available, otherwise Asia/Kolkata. Time slots retain milliseconds.

No face, a large face-box change, or a gap longer than 0.75 seconds resets continuity and
silences the alarm. After such an interruption, observed closure must pass the alert delay again before sound resumes within the same unfinished episode. Missing observations do not count as closed eyes or confirmed reopening.
An episode whose end was never observed is reported as `Ongoing / end not observed`.
An episode crossing midnight belongs to its start date, avoiding double-counting.

### Browser sound and location

Click **Enable browser alarm** inside the browser controls, then START the camera. That
click unlocks Web Audio under browser autoplay rules. Sound follows closed/open episode
state; opening the eyes stops sound without waiting for the 0.5-second episode recovery.
The frontend receives state updates every 0.1 seconds; actual latency depends on camera,
network, and inference speed. A browser watchdog silences audio if updates stall for
0.7 seconds, the tab becomes hidden, or navigation occurs. Normal blinks do not trigger audio.
This uses the browser's speakers, rather than the server's sound device.

Click **Allow current location** to request the browser's geolocation permission. A fix
includes latitude, longitude, accuracy in meters, and acquisition time. Fixes older than
60 seconds or invalid coordinates are treated as unavailable. **Stop location sharing**
revokes the app's active location watch and clears its latest fix; browser permission itself
can also be revoked in browser settings. Location denial never blocks video or TEST delivery.
HTTPS or localhost is required. The host's iframe/Permissions-Policy must allow geolocation.

### Consent and screenshot deletion

Screenshot consent is off by default. If checked when an episode starts, exactly one JPEG
is saved for that episode in `.monitor/events.sqlite3`. No further frames are captured for
that episode, including after brief reopenings. Enabling consent after an episode has started
does not retroactively capture it. The **Consented screenshots** tab can delete one image or
all images belonging to this browser session and driver profile. Deleting a screenshot does
not remove the episode or cause recapture. No webcam video is stored.

SQLite rows are scoped by a random Streamlit session ID plus driver profile. This is session
isolation, not login authentication; browser refresh/session loss can make old session history
unavailable from the UI. Administrators with filesystem access can clear `.monitor/` when
monitoring is stopped. Cloud local disks may be ephemeral; use persistent authenticated
storage before treating this as a durable multi-user deployment.

The desktop command also uses the episode engine. It saves no screenshots by default:

```bash
python drowsiness_detect.py --night --consent-screenshots --driver "Driver 7"
python drowsiness_detect.py --driver "Driver 7" --delete-screenshots
```

The old `screenshots/*.png` files are historical captures from the previous implementation.
They are retained. New desktop consented screenshots use the SQLite store and the deletion
command above; browser screenshots use their session-scoped controls.

### Minimal daily reports

The **Reports** tab, downloaded PDF, and episode CSV contain only:

- Date.
- Start/end time slots for episodes starting on that date.
- Total number of those episodes.

There are no report charts, screenshot images, rest advice, driver details, location fields,
or ML scores. Model Insights and upload diagnostics remain separate from the daily report.
Recorded videos use media timestamps and the same episode rules, but never send live TEST
messages or automatically save screenshots. Their diagnostics do not populate real trip history.

### TEST messaging configuration and delivery status

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` locally, or paste equivalent
values into the deployed Streamlit app's Secrets. Supply a Telegram bot token and an authorized
**test chat_id**. The recipient is configuration, not a hardcoded police/emergency number.
All messages are permanently prefixed **TEST** in this implementation. No real message is sent
by the automated tests. Enable **TEST alerts** in the sidebar before monitoring to authorize
sending to that configured test recipient. Leave the deployed app private while testing.

At **10 seconds of continuous observed closure**, the app queues one logical message per
episode, with driver ID/name, UTC trigger time, continuous duration, and a Google Maps link
plus browser-reported location accuracy. If location is missing, the message says so and
is still queued. A brief reopening resets this 10-second clock, even within the same episode.

A daemon worker performs network I/O outside the video callback. Queue identity is the
unique episode ID, so continued closure does not enqueue another message. Delivery status
(`queued`, `sending`, `retrying`, `sent`, `failed`, or `delivery_unknown`) is shown next to
monitoring. Explicit transient Telegram errors are retried up to three attempts with backoff;
rate-limit retry delays are honored. Configuration errors do not retry indefinitely.

A timeout can occur after a provider accepted a message. Such ambiguous failures become
`delivery_unknown` and are **not automatically replayed**; inspect the TEST chat before taking
any further action. Telegram does not provide an end-to-end exactly-once send guarantee:
application deduplication guarantees one logical job per episode, not certainty about provider
behavior after an ambiguous failure. A worker/app crash during `sending` likewise requires
manual chat inspection; do not blindly resend. This application is a test notifier, not an
emergency dispatch service.

The local SQLite outbox retains retry jobs during the process lifetime. After an app restart,
queued work is resumed when TEST monitoring enqueues another event; do not rely on a sleeping
Streamlit host for unattended delivery. Production delivery requires a durable external worker.

For deployed camera connectivity, configure `rtc_configuration.iceServers` in Streamlit
secrets. The app now reads hosted secrets directly, without requiring a local secrets file.
Some networks require authenticated TURN as well as STUN.

### Verification and manual deployment checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
node tests/test_browser_controls.js
```

Episode tests use fake clocks, mocked messaging, and mocked location. Browser-control unit
tests use a detached JavaScript harness with fake Web Audio/geolocation; they do not grant
real device permissions. PDF tests inspect extracted text and verify no images are embedded.
The report layout is also rendered and visually reviewed with a test-only fixture.

Check these manually on the actual HTTPS deployment:

1. Install `requirements.txt` in Python 3.11/3.12; verify dlib and the landmark/model files load.
2. START the browser camera and verify day/night delay, blink silence, and immediate observed-open sound stop.
3. Enable browser audio and verify audible output and fail-silent behavior on STOP/tab hiding/disconnect.
4. Accept and deny location permission; inspect the TEST message's Maps link, accuracy, or unavailable text.
5. Configure your own TEST chat and verify a 10-second event, delivery status, and provider behavior. No live send is performed by the test suite.
6. Verify one consented screenshot, deletion, and the minimal PDF/CSV time slots/count.
7. Check host STUN/TURN and iframe geolocation policy, CPU lag, and local storage persistence.

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

## Streamlit Community Cloud deployment

The default **HTTPS camera** sends small JPEG frames through Streamlit's existing
connection, so it does not require a separate STUN/TURN media connection. Its
local preview stays smooth; analysis runs at up to five frames per second and
waits for each server acknowledgement before capturing another frame. Actual
analysis rate depends on network latency and server load. Delayed frames are
rejected; gaps stop the alarm and reset continuous-closure timing. Keep the tab
visible and use **Stop camera** to release the device. Live images are temporary;
only explicitly consented episode screenshots are stored.

The optional **WebRTC** mode remains available. A STUN-only configuration can
time out on Community Cloud or restricted networks; configure an authenticated
TURN server in Streamlit Secrets to use that mode on those networks. The
account/avatar 404 messages from Streamlit's hosting interface are independent
of the application's camera transport.

Live app: [Drowsiness Monitor](https://nullv-ctor-drowsiness.streamlit.app/).
Verified on 27 November 2026: cloud build succeeded with Python 3.11,
the trained classifier loaded, real evaluation metrics appeared, and the
Reports page plus browser camera/alarm/location controls rendered.
Physical camera, audible alarm, GPS, and Telegram delivery still require
device checks and your own test-recipient credentials.

Deploy repository `NullV-ctor/Drowsiness`, branch `main`, entrypoint
`cloud/app.py`, and select **Python 3.11** under Advanced settings.
The entrypoint runs the same application as `streamlit_app.py`.
`cloud/requirements.txt` installs only browser app dependencies, with headless
OpenCV and a prebuilt `dlib-bin` wheel to avoid compiling dlib on the cloud host.
The root requirements remain available for notebook training and desktop scripts.
Streamlit's dependency selection is described in its
[deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

The real trained model is included at `models/eye_state_model.npz`.
`experiments/deployment_metrics.json` is a copy of the executed notebook's
metrics, checked against that model's SHA-256. Model Insights uses it when
the local ignored `reports/metrics.json` is absent. Training images are not
needed on the hosting server.

Set actual credentials in the cloud app's Secrets panel using
`.streamlit/secrets.example.toml` as a reference; never use its placeholder
values as real credentials. Telegram TEST messaging needs your own authorized
test recipient. Networks that block direct WebRTC need authenticated TURN.
Cloud SQLite storage is temporary, and browser session history is not a login
system. Durable storage and authenticated accounts remain separate work.

After deployment, check the HTTPS app, camera permissions and video connection,
browser alarm, location allow/deny flows, screenshot consent/deletion, and PDF
download on a real device. No live deployment or device check is implied by
local automated tests.

## Original scripts

Run from the project folder in the installed environment:

```bash
python face_and_eye_detector_single_image.py
python face_and_eye_detector_webcam_video.py
python drowsiness_detect.py
```

The desktop detector uses EAR and plays `audio/alert.wav`. Both desktop and browser monitoring save one episode screenshot only with explicit consent; see the episode section for deletion controls.

## Verification

```bash
python -m unittest discover -s tests -v
```

The checks cover closure timing, missing-face and gap resets, face changes, feature consistency, exported model parity, subject isolation, source-label mapping, data conflicts, training-only augmentation, and matched-method validation/test separation. Real model quality must be measured on an independent labeled dataset. Glasses, lighting, pose, and crop differences can affect predictions. This is an experimental eye-closure project.

Original project: [mohitwildbeast/Driver-Drowsiness-Detector](https://github.com/mohitwildbeast/Driver-Drowsiness-Detector). EAR implementation inspired by Adrian Rosebrock's PyImageSearch tutorial.

MRL data source: [MRL Eye Dataset](https://mrl.cs.vsb.cz/eyedataset.html), MRL / Radovan Fusek. Grouped validation follows [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
