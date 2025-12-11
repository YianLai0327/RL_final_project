import argparse
import json
import os
import tempfile
from pathlib import Path

import laion_clap
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from moviepy import VideoFileClip
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VideoEncoder:
    def __init__(self, device=DEVICE, imagebind=None):
        self.device = device
        if imagebind is None:
            self.imagebind = self._load_imagebind()
        else:
            self.imagebind = imagebind

    def _load_imagebind(self):
        print("Loading ImageBind...")
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(self.device)
        return model

    def encode_segment(self, video_path):
        """
        Extracts segment and encodes with both models.
        """
        # 1. Encode with ImageBind (Video modality)
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(
                [video_path], self.device, clip_duration=2, clips_per_video=5
            ),
        }
        with torch.no_grad():
            ib_embedding = self.imagebind(inputs)[ModalityType.VISION]  # [1, 1024]
            assert isinstance(ib_embedding, torch.Tensor)
            assert ib_embedding.shape == (
                1,
                1024,
            ), f"Expected shape (1, 1024), got {ib_embedding.shape}"

        return {"imagebind": ib_embedding.squeeze().cpu()}


class MusicEncoder:
    def __init__(self, device=DEVICE, imagebind=None, clap=None):
        self.device = device
        if imagebind is None:
            self.imagebind = self._load_imagebind()
        else:
            self.imagebind = imagebind
        if clap is None:
            self.clap = self._load_clap()
        else:
            self.clap = clap

    def _load_imagebind(self):
        # Re-use or load fresh
        print("Loading ImageBind for Music...")
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(self.device)
        return model

    def _load_clap(self):
        print("Loading CLAP...")
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        # Download checkpoint if not exists
        ckpt_path = ".checkpoints/music_audioset_epoch_15_esc_90.14.pt"
        if not os.path.exists(ckpt_path):
            os.makedirs(".checkpoints", exist_ok=True)
            url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
            torch.hub.download_url_to_file(url, ckpt_path)
        model.load_ckpt(ckpt_path)
        model.to(self.device)
        return model

    def encode_file(self, audio_path):
        """
        Encodes the entire audio file.
        """
        # 1. Encode with ImageBind (Audio modality)
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                [audio_path], self.device
            ),
        }
        with torch.no_grad():
            ib_embedding = self.imagebind(inputs)[ModalityType.AUDIO]
            assert isinstance(ib_embedding, torch.Tensor)
            assert ib_embedding.shape == (
                1,
                1024,
            ), f"Expected shape (1, 1024), got {ib_embedding.shape}"

        # 2. Encode with CLAP
        # CLAP supports audio files directly or arrays
        with torch.no_grad():
            # get_audio_embedding_from_file_list expects a list of paths
            clap_embedding = self.clap.get_audio_embedding_from_filelist(
                x=[audio_path], use_tensor=True
            )
            assert isinstance(clap_embedding, torch.Tensor)
            assert clap_embedding.shape == (
                1,
                512,
            ), f"Expected shape (1, 512), got {clap_embedding.shape}"

        return {
            "imagebind": ib_embedding.squeeze().cpu(),
            "clap": clap_embedding.squeeze().cpu(),
        }


def parse_time_str(time_str) -> float:
    """Converts MM:SS string to seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


def process_videos(data_dir, encoder, dry_run=False, force=False):
    json_path = os.path.join(data_dir, "video_captions.json")
    video_dir = os.path.join(data_dir, "raw_videos")
    output_dir = os.path.join(data_dir, "video_embs")

    if not os.path.exists(json_path):
        print(
            f"Warning: {json_path} not found. Skipping video processing for {data_dir}."
        )
        return

    if not os.path.exists(video_dir):
        print(
            f"Warning: {video_dir} not found. Skipping video processing for {data_dir}."
        )
        return

    print(f"Processing videos in {data_dir}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Videos in {os.path.basename(data_dir)}"):
        filename = item["filename"]
        video_path = os.path.join(video_dir, filename)

        if not os.path.exists(video_path):
            print(f"  [Warning] Video not found: {filename}")
            continue

        if dry_run:
            print(f"  [Dry Run] Would create dir and process: {filename}")
            continue

        vid_out_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        os.makedirs(vid_out_dir, exist_ok=True)

        # Load video
        clip = VideoFileClip(video_path)

        pbar = tqdm(item.get("segments", []), desc=f"  Segs {filename}", leave=False)
        for i, segment in enumerate(pbar):
            save_path = os.path.join(vid_out_dir, f"segment_{i}.pt")
            if os.path.exists(save_path) and not force:
                continue

            start_time = max(0.0, parse_time_str(segment["start"]))
            end_time = min(clip.end, parse_time_str(segment["end"]))

            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp_path = tmp.name

                # Subclip video
                clip.subclipped(start_time, end_time).write_videofile(
                    tmp_path, temp_audiofile_path=tempfile.gettempdir(), logger=None
                )

                embeddings = encoder.encode_segment(tmp_path)
                if embeddings:
                    torch.save(embeddings, save_path)
        clip.close()


def process_music(data_dir, encoder, dry_run=False, force=False):
    json_path = os.path.join(data_dir, "music_captions.json")
    music_dir = os.path.join(data_dir, "raw_music")
    output_dir = os.path.join(data_dir, "music_embs")

    if not os.path.exists(json_path):
        print(
            f"Warning: {json_path} not found. Skipping music processing for {data_dir}."
        )
        return

    if not os.path.exists(music_dir):
        print(
            f"Warning: {music_dir} not found. Skipping music processing for {data_dir}."
        )
        return

    print(f"Processing music in {data_dir}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Music in {os.path.basename(data_dir)}"):
        filename = item["filename"]
        music_path = os.path.join(music_dir, filename)

        if not os.path.exists(music_path):
            print(f"  [Warning] Music not found: {filename}")
            continue

        if dry_run:
            print(f"  [Dry Run] Would process: {filename}")
            continue

        mus_out_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        os.makedirs(mus_out_dir, exist_ok=True)

        save_path = os.path.join(mus_out_dir, "full.pt")
        if os.path.exists(save_path) and not force:
            continue

        embeddings = encoder.encode_file(music_path)
        torch.save(embeddings, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess video and music embeddings."
    )
    parser.add_argument(
        "dirs", nargs="*", help="Directories to process (e.g. small, medium)."
    )
    parser.add_argument(
        "--process_video", action="store_true", help="Process video segments."
    )
    parser.add_argument(
        "--process_music", action="store_true", help="Process full music tracks."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write files, just simulate."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite existing files."
    )

    args = parser.parse_args()

    dirs_to_process = args.dirs
    if not dirs_to_process:
        current_dir = Path(".")
        candidates = ["small", "medium"]
        dirs_to_process = [d for d in candidates if (current_dir / d).is_dir()]

        if not dirs_to_process:
            print(
                "No directories provided and 'small'/'medium' not found. searching all subdirectories..."
            )
            dirs_to_process = [x.name for x in current_dir.iterdir() if x.is_dir()]

    video_encoder = None
    music_encoder = None

    if not args.dry_run:
        if args.process_video:
            print("Initializing Video Encoder...")
            video_encoder = VideoEncoder()

        if args.process_music:
            print("Initializing Music Encoder...")
            if video_encoder is not None:
                # Share imagebind model
                music_encoder = MusicEncoder(imagebind=video_encoder.imagebind)
            else:
                music_encoder = MusicEncoder()

    for d in dirs_to_process:
        d_path = os.path.abspath(d)
        if not os.path.exists(d_path):
            print(f"Directory {d} does not exist. Skipping.")
            continue

        print(f"=== Processing Directory: {d} ===")

        if args.process_video:
            process_videos(
                d_path, video_encoder, dry_run=args.dry_run, force=args.force
            )

        if args.process_music:
            process_music(d_path, music_encoder, dry_run=args.dry_run, force=args.force)

        print("\n")


if __name__ == "__main__":
    main()
