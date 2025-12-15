import argparse
import math
import os
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rl_soundtrack.dataset.discrete import DiscreteDataset
from rl_soundtrack.utils.common import load_config
from tqdm import tqdm


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def truncate(filename, max_length=20):
    if len(filename) > max_length:
        return filename[:max_length] + "..."
    return filename


def calc_video_ref_embedding(dataset):
    per_video_avg_embeddings = []
    for video in dataset.videos:
        if not video.segments:
            continue

        video_segment_embeddings = []
        for segment in video.segments:
            video_segment_embeddings.append(segment.caption_embedding)

        # Per-video average
        per_video_avg_embeddings.append(np.mean(video_segment_embeddings, axis=0))

    # Across-video average
    return normalize(np.mean(per_video_avg_embeddings, axis=0))


def compute_similarity(video_emb, music_embs):
    """
    video_emb: (D,)
    music_embs: (N_music, D)
    Returns: (N_music,) cosine similarities
    """
    # content is already numpy flat array
    v = normalize(video_emb)
    # music_embs should satisfy: valid 2D array
    m = music_embs  # assume already normalized for efficiency

    return np.dot(m, v)


def analyze_video_similarity(video, v_ref_emb, music_embs_matrix):
    """
    Computes similarities for all segments of a video against all music.
    Returns:
        stats dict containing lists of max, mean, min similarities over segments.
    """
    max_sims = []
    mean_sims = []
    min_sims = []
    std_sims = []
    all_sims = []

    for seg in video.segments:
        # Get video embedding
        # seg.caption_embedding is already preloaded by load_embeddings()
        if seg.caption_embedding is not None and seg.caption_embedding.shape[0] > 0:
            v_emb = seg.caption_embedding
        else:
            v_emb = np.zeros(1024, dtype=np.float32)

        sims = compute_similarity(v_emb, music_embs_matrix)
        ref_sims = compute_similarity(v_ref_emb, music_embs_matrix)
        sims = sims - ref_sims

        max_sims.append(np.max(sims))
        mean_sims.append(np.mean(sims))
        min_sims.append(np.min(sims))
        std_sims.append(np.std(sims))
        all_sims.append(sims)

    return {
        "max": max_sims,
        "mean": mean_sims,
        "min": min_sims,
        "std": std_sims,
        "all": all_sims,
    }


def plot_similarities(all_video_results, dataset, save_dir):
    save_path = pathlib.Path(save_dir).joinpath("dcaption_similarity_analysis.png")

    num_videos = len(all_video_results)
    if num_videos == 0:
        print("No videos to plot.")
        return

    num_rows = math.ceil(math.sqrt(num_videos))
    # Adjust aspect ratio if too wide/tall
    num_cols = num_rows
    if num_videos <= num_rows * (num_rows - 1):
        num_rows -= 1

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6 * num_cols, 4 * num_rows),
        sharey=True,
    )

    if num_videos == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Shared Y limits for cosine similarity usually [-1, 1], but for ImageBind roughly [0, 1] or tighter
    # We'll stick to automatic or fixed [-1, 1]

    for i, result in enumerate(all_video_results):
        if i >= len(axes):
            break
        ax = axes[i]
        filename = result["filename"]
        truncated_filename = filename if len(filename) <= 30 else filename[:27] + "..."
        ax.set_title(f"{truncated_filename}")
        ax.set_xlabel("Segment Index")
        ax.set_ylabel("Caption Cosine Similarity")

        stats = result["stats"]
        x = range(len(stats["max"]))

        ax.scatter(x, stats["max"], label="Max Sim (Best)", alpha=0.5)
        ax.plot(x, stats["mean"], label="Mean Sim", linestyle="--")
        ax.scatter(x, stats["min"], label="Min Sim", alpha=0.5)

        all_sims = np.array(stats["all"])
        mean_sims = np.mean(all_sims, axis=0)  # (n_music,)
        top_indices = np.argsort(mean_sims)[-3:][::-1]
        top_filenames = [truncate(dataset.tracks[i].filename) for i in top_indices]
        bottom_indices = np.argsort(mean_sims)[:3]
        bottom_filenames = [
            truncate(dataset.tracks[i].filename) for i in bottom_indices
        ]
        ax.plot(all_sims[:, top_indices], label=top_filenames)
        ax.plot(all_sims[:, bottom_indices], label=bottom_filenames)

        # Fill between mean +/- std
        mean = np.array(stats["mean"])
        std = np.array(stats["std"])
        ax.fill_between(
            x, mean - std, mean + std, color="blue", alpha=0.1, label=r"Mean $\pm$ Std"
        )

        ax.grid(True)
        # Legend only on first plot to save space, or all if sparse
        # if i == 0:
        ax.legend(loc="lower center", ncols=2)

        # ax.set_ylim(-0.3, 0.9)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    print(f"Saving similarity plot to {save_path}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def print_top_matches(all_video_results, dataset):
    print("\n--- Top 3 Music Matches (Mean Similarity) ---")
    # Header
    print(f"{'Video':<30} | {'Top 1':<35} | {'Top 2':<35} | {'Top 3':<35}")
    print("-" * 145)

    for result in all_video_results:
        filename = result["filename"]
        truncated_video = filename if len(filename) <= 28 else filename[:25] + "..."

        all_sims = np.array(result["stats"]["all"])  # (n_segments, n_music)
        if all_sims.size == 0:
            continue

        # Mean over segments
        mean_sims = np.mean(all_sims, axis=0)  # (n_music,)

        # Get top 3
        top_indices = np.argsort(mean_sims)[-3:][::-1]

        row_str = f"{truncated_video:<30}"

        for idx in top_indices:
            score = mean_sims[idx]
            track_name = dataset.tracks[idx].filename
            truncated_track = (
                track_name if len(track_name) <= 20 else track_name[:17] + "..."
            )
            entry = f"{truncated_track} ({score:.3f})"
            row_str += f" | {entry:<35}"

        print(row_str)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Cosine Similarity (ImageBind) between Videos and Music Library"
    )
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        default="data/medium",
        help="Path to dataset dir",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_dir}...")
    # Initialize dataset
    dataset_path = Path.cwd().joinpath(args.dataset_dir)
    # The dataset class deduces paths relative to CWD usually, but we can pass exact logic if needed.
    # checking discrete.py: data_dir logic handles absolute paths if provided or defaults.
    # If args.dataset_dir is relative to cwd, we should probably resolve it or rely on DiscreteDataset logic.
    # DiscreteDataset defaults to CWD/data if not provided. If provided, uses it.

    dataset = DiscreteDataset(data_dir=str(dataset_path))

    # 1. Prepare Music Matrix
    print("Preparing music embeddings matrix...")
    music_matrix = []

    valid_tracks = 0
    for track in dataset.tracks:
        if track.caption_embedding is not None and track.caption_embedding.shape[0] > 0:
            music_matrix.append(normalize(track.caption_embedding))
            valid_tracks += 1
        else:
            # Handle missing embeddings
            music_matrix.append(np.zeros(1024, dtype=np.float32))

    music_matrix = np.array(music_matrix)  # Shape (N, D)
    print(f"Music Matrix Shape: {music_matrix.shape}")

    all_video_results = []

    print("Analyzing videos...")
    video_ref_emb = calc_video_ref_embedding(dataset)
    for video in tqdm(dataset.videos):
        stats = analyze_video_similarity(video, video_ref_emb, music_matrix)
        all_video_results.append({"filename": video.filename, "stats": stats})

    # Sort by filename for consistent plotting
    all_video_results.sort(key=lambda x: x["filename"])

    print("Plotting...")
    plot_similarities(all_video_results, dataset, str(dataset_path))

    print_top_matches(all_video_results, dataset)

    print("Done!")


if __name__ == "__main__":
    main()
