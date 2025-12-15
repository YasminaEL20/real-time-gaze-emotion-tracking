# real-time-gaze-emotion-tracking

Real-time gaze tracking and facial emotion recognition (computer vision).

Overview
-
This project is a real-time computer-vision application that simultaneously
tracks user gaze/focus and recognizes facial emotions from a webcam feed.
It combines a face detection + landmark pipeline (MediaPipe/OpenCV) with a
pre-trained Keras/TensorFlow emotion classifier (provided as
`fer2013_emotion_model.h5`). The system annotates the live video stream,
logs gaze events to CSV, and exposes utilities for visualization and further
analysis.

How it works (high level)
-
- Capture frames from the webcam.
- Detect face and facial landmarks using MediaPipe.
- Estimate gaze direction/point-of-regard from facial landmarks and pupil position.
- Crop/prepare face region and run the emotion model to classify emotion labels
  (e.g., happy, sad, angry, neutral).
- Render annotations (face box, gaze point, emotion) and append gaze events to
  `gaze_data.csv` with timestamps and metadata.

Use cases
-
- Research and prototyping for attention and affect analysis.
- UX testing (where users look and how they react).
- Educational demos combining computer vision and affective computing.

Why Python 3.8 / 3.10 / 3.11 (recommended) and not the very newest versions
-
- Binary compatibility: many ML and CV packages (TensorFlow, MediaPipe,
  OpenCV) publish pre-built wheels for stable Python releases first. At the
  moment, official and widely used wheels are most reliably available for
  Python 3.8–3.11. Using these versions reduces the chance of installation
  failures or needing to build packages from source.
- Stability and ecosystem support: package maintainers test extensively on
  the commonly used language versions above. Newer Python releases may
  introduce ABI or packaging changes that break pre-built binaries for a
  short period after release.
- Feature balance: Python 3.10 and 3.11 include useful language/performance
  improvements over 3.8 while remaining widely supported by ML/CV libraries.

Recommendation: prefer `3.10` or `3.11` where possible. Use `3.8` if you
have specific compatibility constraints. Avoid brand-new Python releases
unless you confirm that all required wheels are available for that version.

Creating and activating a virtual environment (detailed) :
-
You can create a virtual environment with any local Python executable you have
downloaded. Choose a name for the environment (for example `.venv`, `env`, or
any name you prefer). Below are common Windows examples.

- Using the `py` launcher to pick a specific Python version (recommended on Windows):
-checking your ptyhon version in the termina :
    python --version
    "yous must have the version witch you have installed"

and do :

  ```powershell
  # Create a venv using Python 3.10
  py -3.10 -m venv .venv

  # Or create a venv using Python 3.11
  py -3.11 -m venv myenv
  ```
  it depend to your python version

- Using an absolute Python executable (if you installed a specific Python yourself):

  ```powershell
  # Replace the path with the python.exe you downloaded
  &C:\Path\To\Python310\python.exe -m venv .venv
  ```
    execute :
        where python

- Activation (PowerShell):

  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- Activation (Command Prompt):

  ```cmd
  .venv\Scripts\activate.bat
  ```

- Activation (Git Bash / WSL):

  ```bash
  source .venv/Scripts/activate
  ```

Notes:
- You may name the folder anything (`.venv`, `myenv`); `.venv` is a common
  convention that many tools ignore automatically.
- If PowerShell refuses to run the activation script due to execution policy,
  run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
  from an elevated PowerShell (or use Command Prompt activation instead).

Upgrade pip and essential build tools
-
After activating the venv, upgrade `pip`, `setuptools` and `wheel` to get the
most reliable installation behavior and wheel support:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Installing dependencies from `requirements.txt`
-
Ensure you have a stable internet connection (some packages are large). Then:

```bash
pip install -r requirements.txt
```

Notes and reminder:
- The repository includes a `requirements.txt` file (not CSV). If you see
  a filename like `requirements.csv` that is a mistake, use the `requirements.txt`.
- Downloads may take several minutes depending on connection speed. Be patient
  and avoid interrupting the install.

Running the application
-
The repository entry point is `main_tracker.py`. Run it from the activated
virtual environment:

```bash
python main_tracker.py
```

If you have a different entry script named `main.py` (some users rename files),
you can run:

```bash
python main.py
```

During execution a window will show live annotations. Gaze events are appended
to `gaze_data.csv` in the repository folder for later analysis.

Quick troubleshooting
-
- Camera issues: try a different camera index (look for `cv2.VideoCapture(0)`),
  or test the camera with a simple script.
- Model loading errors: ensure `fer2013_emotion_model.h5` is present in the
  project root and that you installed a compatible `tensorflow`/`keras` version.
- Long installs: if `pip install -r requirements.txt` fails, capture the
  error and consult the package name/version that failed.

Contributing
-
- Open issues for bugs or feature requests and submit pull requests for fixes.

License & contact
-
Specify your license here (e.g., MIT) and provide contact details if desired.

---
If you want, I can now commit these README changes and push them to `origin main`.
