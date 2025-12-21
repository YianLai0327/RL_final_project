# split_song

## Build environment ✅
Recommended: Python 3.10

Install required packages using pip:

```bash
# It is recommended to create an isolated environment (conda or venv)
conda create -n split-song python=3.10 -y
conda activate split-song

# Install TensorFlow (for GPU support, make sure CUDA / cuDNN and drivers are installed and compatible)
pip install "tensorflow[and-cuda]"

# Install audio / feature analysis and other dependencies
pip install openl3 ruptures soundfile numpy
```

💡 Note: GPU support for TensorFlow requires the correct CUDA and cuDNN versions and matching GPU drivers installed on your system.

---

## Data 📂
Data is stored in `split_song/split.json`. The JSON format is:

- key: video title (or filename)
- value: an array where each object contains:
  - `time`: detected timestamp (seconds)
  - `score`: confidence score for a "switch song" boundary (higher means more likely)

Example (excerpt):
```json
{
  "VIDEO_TITLE": [
    {"time": 19.0, "score": 0.012068629264831543},
    {"time": 38.0, "score": 0.019410908222198486}
  ]
}
```

- Larger `score` values indicate higher confidence that the timestamp is a song-change boundary.
- From listening tests, a practical threshold for detecting song changes is `0.02` (i.e., consider `score >= 0.02` as a switch).
  - This is an empirical starting point and not guaranteed to be optimal.
  - To improve precision, validate and tune the threshold across samples or add post-processing rules (e.g., minimum distance between events, smoothing, or multi-criteria filtering).

---