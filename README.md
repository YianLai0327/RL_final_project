# Team 12 Final project

## Abstract
This repository implements **RL-Soundtrack: Reinforcement Learning for Video-Aware Sequential BGM Recommendation**. The rest of this README provides step-by-step instructions for environment setup, data collection and preprocessing, model training, and evaluation/inference.

---

## Folder Structure
Below is a concise description of the main folders in this repository. Some folders contain their own `README.md` files — please see those for detailed instructions.

- `rl-soundtrack/` 🔧
  - Core RL codebase: environments, agents, dataset utilities, and preprocessing scripts.
  - See `rl-soundtrack/README.md` for setup, preprocessing, training, and evaluation instructions.

- `reward_analysis/` 🎯
  - Experimental scripts to run interactive/manual soundtrack selection and compute evaluation artifacts (e.g., `confusion_matrix.png`).
  - Contains utilities like `split_mp3.py` and `auto_action.py` and a `README.md` explaining the data preparation workflow.

- `split_song/` ✂️
  - Tools for detecting song-change points and generating `split.json`. See `split_song/README.md` for data format and thresholds.

- `caption_final/` 📝
  - Pipelines and scripts for generating audio/video captions (Gemini / Google GenAI based). Most captioning scripts and helper utilities live here.
  - `caption_final/README.md` documents the caption generation steps and requirements.

- `crawl_vlog/` 🌐
  - Scripts for crawling/downloading vlog videos and audio, plus utilities to make JSON datasets used by captioning and downstream steps.

- `BGM/` 🎵
  - Background music collection scripts, chorus extraction, and audio feature extraction code.

---

## Environments
Follow the [rl-README.md](rl-soundtrack/README.md) instructions for an environment setup; here is an adapted summary:

1. Create and activate a Conda environment (recommended):

```bash
conda create -n rl-soundtrack python=3.12 -y
conda activate rl-soundtrack
```

2. Install system dependencies (example):

```bash
sudo apt install ffmpeg
# Install Python deps
pip install -r rl-soundtrack/requirements.txt
pip install -e rl-soundtrack
```

3. GPU notes: if you need GPU acceleration (TensorFlow / PyTorch), make sure CUDA and cuDNN are installed and compatible with your selected packages.

---

## Preprocess / Data Collection
There are two ways to prepare data for experiments:

1. Quick / lazy approach (recommended for rapid testing):
   - Ensure `gdown` is installed (`pip install gdown`), then run the repository script to download datasets:
   ```bash
   # From repository root
   bash download_dataset.sh
   ```
   - After downloading, enter the `rl-soundtrack/` directory and run the dataset format & embedding scripts:
   ```bash
   cd rl-soundtrack
   python data/format_dataset.py data/<dataset_name>
   python data/calculate_embs.py --process_video --process_music data/<dataset_name>
   ```
   - This produces `music_captions.json`, `video_captions.json`, embeddings, and other artifacts ready for training/eval.

2. Build your own dataset from scratch:
   - Use `crawl_vlog/` to download raw vlog media and generate JSON crawls.
   - Use `caption_final/` (or `caption/`) to create Gemini captions for audio and video (`audio_captioning.py`, `video_captioning.py`).
   - Use `BGM/` scripts to collect and preprocess background music and extract features.
   - Use `split_song/` to detect and/or manually author `split.json` with song-change timestamps, and `split_mp3.py` to produce segmented mp3 files.
   - Finally, format and compute embeddings with `rl-soundtrack` as above.

---

## Eval / Inference
Follow the `rl-soundtrack/README.md` evaluation commands. Adapted example:

```bash
# Evaluate a trained model (example)
cd rl-soundtrack
python scripts/eval.py -m models/<model_path> -d data/<dataset_name> -n <episodes> -r
```

- Visualization and summary scripts (e.g., reward matrices, confusion matrices) may live in `reward_analysis/` or `rl-soundtrack/scripts`.

---

## Train
Refer to `rl-soundtrack/README.md` for training commands. Example:

```bash
cd rl-soundtrack
python scripts/train.py -d data/<dataset_name>
# Continue training
python scripts/train.py --id <run_id> -d data/<dataset_name>
```

- Hyperparameters and configs are typically stored under `rl-soundtrack/configs/`.
