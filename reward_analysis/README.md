# reward_analysis

## Overview ✅
This folder contains an experiment pipeline that lets the model automatically assign background music (BGM) to vlogs and evaluates pairings by saving a `confusion_matrix.png` that summarizes rewards for different audio–video combinations. The workflow covers data preparation (splitting audio, generating captions, computing embeddings) and running `auto_action.py` for interactive experiments.

---

## Prerequisites 🔧
- Python 3.8+
- ffmpeg installed (required for audio processing)
- Recommended: use a conda or venv environment
- Gemini / Google GenAI API key is required if you plan to generate captions with the `../caption_final` scripts
- See `../rl-soundtrack/README.md` for instructions on building datasets and computing embeddings

---

## Files 📁
- `split.json`: Describes split timestamps (in seconds) for each vlog (or its corresponding `.mp3`).
- `split_mp3.py`: Reads `split.json`, slices mp3 files from `separated/` and writes segments to `splited/`.
- `auto_action.py`: Interactive script that loads a prepared dataset (formatted via `rl-soundtrack`), runs episodes, and outputs `confusion_matrix.png`.
- `data/`: Place processed dataset files here (music/video libraries, captions, embeddings) for use by `auto_action.py`.

---

## Data Preparation (step-by-step) 🧭
1. Prepare vlog audio (after source separation) and place the mp3 files in:
   - `reward_analysis/separated/` (matches `split_mp3.py`'s `ROOT_DIR`)

2. Create `split.json` where each key is an mp3 filename and the value is a list of timestamps (seconds) indicating split points.

   Example format:
   ```json
   {
     "my_vlog_01.mp3": [30, 90, 200],
     "another_vlog.mp3": [40, 120]
   }
   ```
   - Note: `auto_action.py`'s `split_data()` inserts a leading `0` and removes the list's maximum value. Include the video end time (or last split point) in `split.json` to ensure correct behavior.

3. Slice the mp3 files into segments:
   ```bash
   cd reward_analysis
   python split_mp3.py
   # Reads from separated/ and writes segments such as <original>_0.mp3 into splited/
   ```

4. Generate Gemini captions (semantic captions) for music and video:
   - Follow `../caption_final/README.md` — use `audio_captioning.py` and `video_captioning.py` to create captions.
   - These scripts typically produce legacy-format files: `music_captions_old.json` (list) and `video_captions_old.json` (dict).
   - If you already have legacy files, place them under `rl-soundtrack/data/<your_dataset>/` for formatting.

5. Format data and compute embeddings (in `rl-soundtrack`):
   ```bash
   cd ../rl-soundtrack
   ```
   Put generated or existing captions and media into data/<dataset_name>/ (see [rl-README](../rl-soundtrack/README.md))
   ```
   python data/format_dataset.py data/<dataset_name>
   python data/calculate_embs.py --process_video --process_music data/<dataset_name>
   ```
   - `format_dataset.py` will read `music_captions_old.json` and `video_captions_old.json`, producing `music_captions.json`, `video_captions.json`, and filename maps.
   - `calculate_embs.py` computes video/music embeddings required to build the environment.

6. Back in `reward_analysis`, ensure `data/` (or a copy of the `rl-soundtrack` dataset) contains `music_library`, `video_library`, captions, and embeddings.

7. Run the experiment:
   ```bash
   cd reward_analysis
   python auto_action.py
   # Optional args: --video-idx, --w-alignment, --w-smoothness, --w-switch, --w-theme
   ```
   - `auto_action.py` runs interactive episodes for multiple (video_idx, audio_idx) combinations and saves `confusion_matrix.png`.

---

## Common Issues / Gotchas ⚠️
- Keys in `split.json` must match the video filenames used by the environment (i.e., `Path(video.filename).stem + '.mp3'`). If not found, `auto_action.py` may fail to retrieve splits.
- If `format_dataset.py` cannot find files, check `video_library/` and `music_library/` filenames; fuzzy matching may help.
- When generating captions with Gemini / Google GenAI, verify that your API key and `caption_final` environment settings are correct.

---

## Resources 🔗
- Caption pipeline: `caption_final/README.md`
- Dataset & preprocessing: `rl-soundtrack/README.md`
- See `split_mp3.py` and `auto_action.py` in this directory for implementation details.

