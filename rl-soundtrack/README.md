# RL-Soundtrack

Reinforcement Learning for Video-Aware Sequential BGM Recommendation.

## Setup

All the commands are run in the `rl-soundtrack` directory.

1. Create Conda environment:

   ```bash
   conda create -n rl-soundtrack python=3.12
   conda activate rl-soundtrack
   ```

2. Install dependencies:

   ```bash
   sudo apt install ffmpeg
   # we test on ffmpeg version 6.1.1-3ubuntu5

   pip install -r requirements.txt
   pip install -e .
   ```

## Folder Structure

```
rl-soundtrack/
├── rl_soundtrack   # Source code.
│   ├── agents      # RL agents.
│   ├── envs        # Custom Gym environments.
│   └── utils       # Helper functions.
├── configs         # Configuration files.
├── data            # Datasets & preprocess scripts.
└── scripts         # Scripts for training and evaluation.
```

## Preprocess

1. Download raw videos and music from [Google Drive](https://drive.google.com/drive/folders/1DCtcJoANZGf51l2wRnLBhO8PnZbknS-p?usp=drive_link), and place them in `data/raw_videos` and `data/raw_music`.

2. Format the dataset:

   ```bash
   python data/format_dataset.py <dataset_dir>
   usage: format_dataset.py [-h] [--dry-run] [dirs ...]

   # example
   python data/format_dataset.py data/small data/medium
   ```

3. Preprocess embeddings:

   ```bash
   python data/calculate_embs.py --process_video --process_music <dataset_dirs>
   usage: calculate_embs.py [-h] [--process_video] [--process_music] [--dry-run] [--force] [dirs ...]

   # example
   python data/calculate_embs.py --process_video --process_music data/small data/medium
   ```

## Train

1. Train the agent:

   ```bash
   python scripts/train.py -d <dataset_dir>
   usage: train.py [-h] [--id ID] [-c CONFIG] [-d DATASET_DIR]

   # example
   python scripts/train.py -d data/medium
   ```

2. Continue training:

   ```bash
   python scripts/train.py --id <run_id> -d <dataset_dir>
   usage: train.py [-h] [--id ID] [-c CONFIG] [-d DATASET_DIR]

   # example (find run_id in models/)
   python scripts/train.py --id 20251208-220756 -d data/medium
   ```

## Evaluate

1. Evaluate the agent:

   ```bash
   python scripts/eval.py -m <model_path> -d <dataset_dir>
   usage: eval.py [-h] [-c CONFIG] -m MODEL [-n EPISODE] [-d DATASET_DIR] [-r]

   # example (with total 47 episodes and rendering)
   python scripts/eval.py -m models/20251208-220756/best_model -n 47 -r -d data/medium
   ```
