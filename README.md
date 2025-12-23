# Group 12 Final project

[[`PDF`](./Report.pdf)] [[`Github`](https://github.com/YianLai0327/RL_final_project)]

## RL-Soundtrack: Reinforcement Learning for Video-Aware Sequential BGM Recommendation

This repository implements **RL-Soundtrack: Reinforcement Learning for Video-Aware Sequential BGM Recommendation**. The rest of this README provides step-by-step instructions for environment setup, data collection and preprocessing, model training, and evaluation/inference.

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
  - Note that you have to use **API KEY** in these section.
  - `caption_final/README.md` documents the caption generation steps and requirements.

- `crawl_vlog/` 🌐

  - Scripts for crawling/downloading vlog videos and audio, plus utilities to make JSON datasets used by captioning and downstream steps.

## Quick Reproduce

Just go to `rl-soundtrack` directory and follow the full instructions in [rl-soundtrack/README.md](rl-soundtrack/README.md).

## Environments

See [Setup](rl-soundtrack/README.md#setup).

_GPU notes:_ if you need GPU acceleration (TensorFlow / PyTorch), make sure CUDA and cuDNN are installed and compatible with your selected packages. In RL, however, GPU is utilized solely for preliminary embedding extraction.

## Execution

### 1. Preprocess / Data Collection

There are two ways to prepare data for experiments:

#### A. Quick / lazy approach (recommended for rapid testing)

See [Quick Download](rl-soundtrack/README.md#quick-download).

#### B. Build your own dataset from scratch

- Use `crawl_vlog/` to download raw vlog media and generate JSON crawls.
- Use `caption_final/` to create Gemini captions for audio and video (`audio_captioning.py`, `video_captioning.py`).
- Use `split_song/` to detect and/or manually author `split.json` with song-change timestamps, and `split_mp3.py` to produce segmented mp3 files.
- Finally, format and compute embeddings with `rl-soundtrack` in [Preprocess](rl-soundtrack/README.md#preprocess).

### 2. Train

See [Train](rl-soundtrack/README.md#train).

### 3. Evaluate

See [Evaluate](rl-soundtrack/README.md#evaluate).
