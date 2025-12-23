# RL-Soundtrack

Reinforcement Learning for Video-Aware Sequential BGM Recommendation.

## Setup

```bash
cd rl-soundtrack
```

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
│   ├── dataset     # Dataset for environment.
│   ├── envs        # Custom Gym environments.
│   └── utils       # Helper functions.
├── configs         # Configuration files.
├── data            # Datasets & preprocess scripts.
└── scripts         # Scripts for training and evaluation.
```

## Dataset

### Quick Download

Manually download dataset zip files from [Google Drive](https://drive.google.com/file/d/1G-fr_DOem0F7SC6OQVWhU7c_1-p310-m/view?usp=sharing), and unzip them. It should create a folder named `data/dataset_198`, and files are put under the folder.

Or run the following script to download and unzip:

```bash
bash download.sh
```

### Preprocess

**[Note]** If you build you own dataset rather than download zip dataset from above, you need to preprocess the embeddings and features.

```bash
python data/preprocess.py --process_video --process_music --label_music data/dataset_198
```

## Train

1. Train the agent:

   ```bash
   python scripts/train.py -d data/dataset_198
   ```

2. Continue training:

   ```bash
   python scripts/train.py -m logs/<model_id>/last_model -d data/dataset_198
   ```

See [scripts/train.py](scripts/train.py) and [configs/default.yaml](configs/default.yaml) for more details. The model will be saved in `logs/<model_id>/` by default.

## Evaluate

1. Evaluate the RL agent with baselines (same as in paper):

   ```bash
   # Training set
   python scripts/eval.py \
      -m logs/<model_id>/ \
      -d data/dataset_198 \
      -n 632 \
      -s train \
      -a rl/best_model \
      -a greedy \
      -a greedy/alignment_reward,same_track_penalty \
      -a random

   # Testing set
   python scripts/eval.py \
      -m logs/<model_id>/ \
      -d data/dataset_198 \
      -n 200 \
      -s test \
      -a rl/best_model \
      -a greedy \
      -a greedy/alignment_reward,same_track_penalty \
      -a random
   ```

The output comparison figures are saved in `logs/<model_id>/eval_comparison` by default.
See [scripts/eval.py](scripts/eval.py) for more details.

More:

- You can use `-r` to render the videos. It takes a lot of time, so try to decrease the number of inferenced videos by `-n <number>` (`158` for full training set, `40` for full testing set).
- You can find output figures/videos in `logs/<model_id>/`.
