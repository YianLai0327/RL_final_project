import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rl_soundtrack.utils.audio_features import compute_audio_features
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class MusicFeatures:
    mfcc: List[float]
    bpm: float
    energy: float
    spectral_centroid: float
    duration: float


@dataclass
class MusicTrack:
    id: int
    filename: str
    metadata: Dict[str, Any]
    imagebind_embedding: Optional[np.ndarray] = None
    caption_embedding: Optional[np.ndarray] = None
    features: Optional[MusicFeatures] = None


@dataclass
class VideoSegment:
    index: int
    metadata: Dict[str, Any]
    imagebind_embedding: Optional[np.ndarray] = None
    caption_embedding: Optional[np.ndarray] = None


@dataclass
class Video:
    id: int
    filename: str
    metadata: Dict[str, Any]
    segments: List[VideoSegment] = field(default_factory=list)


class DiscreteDataset:
    """
    Dataset class to handle loading and storing of music and video data,
    including embeddings and features, using structured dataclasses.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        dataset_name: str = "default",  # New argument for dataset name
        audio_filename: str = "music_captions.json",
        video_filename: str = "video_captions.json",
        text_encoder: Optional[SentenceTransformer] = None,
    ):
        # Path Handling
        if data_dir is None:
            if dataset_name == "default":
                self.data_dir = str(Path.cwd() / "data")
            else:
                self.data_dir = str(Path.cwd() / "data" / dataset_name)
        else:
            self.data_dir = data_dir

        self.audio_path = os.path.join(self.data_dir, audio_filename)
        self.video_path = os.path.join(self.data_dir, video_filename)

        if text_encoder is None:
            self.text_encoder = SentenceTransformer(
                "all-MiniLM-L6-v2", cache_folder=".checkpoints"
            )
        else:
            self.text_encoder = text_encoder

        # Data Storage
        self.tracks: List[MusicTrack] = []
        self.videos: List[Video] = []

        # Initial Load of JSONs
        self._load_initial_data()
        self.n_audio = len(self.tracks)

    def _load_initial_data(self):
        # Load audio data
        audio_data = self._load_json(self.audio_path)
        for i, item in enumerate(audio_data):
            track = MusicTrack(id=i, filename=item.get("filename", ""), metadata=item)
            self.tracks.append(track)

        # Load video data
        video_data = self._load_json(self.video_path)
        for i, item in enumerate(video_data):
            # Only include videos with segments
            if "segments" in item and len(item["segments"]) > 0:
                video_segments = []
                for seg_idx, seg_data in enumerate(item["segments"]):
                    video_segments.append(
                        VideoSegment(index=seg_idx, metadata=seg_data)
                    )

                video = Video(
                    id=i,
                    filename=item.get("filename", ""),
                    metadata=item,
                    segments=video_segments,
                )
                self.videos.append(video)

    def _load_json(self, path: str) -> List[Dict]:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except FileNotFoundError:
            print(f"Warning: Data file not found at {path}. Using empty list.")
            return []

    def _get_emb_path(self, subdir, filename):
        name_no_ext = os.path.splitext(filename)[0]
        return os.path.join(self.data_dir, subdir, name_no_ext)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        emb = self.text_encoder.encode(text, convert_to_numpy=True)[0]
        return emb

    def _read_embedding_from_disk(
        self, subdir, filename, segment_suffix=None
    ) -> np.ndarray:
        emb_dir = self._get_emb_path(subdir, filename)
        if segment_suffix is not None:
            path = os.path.join(emb_dir, f"segment_{segment_suffix}.pt")
        else:
            path = os.path.join(emb_dir, "full.pt")

        if os.path.exists(path):
            try:
                data = torch.load(path, map_location="cpu")
                if isinstance(data, dict):
                    emb = data.get("imagebind")
                else:
                    emb = data

                if emb is not None:
                    return emb.detach().numpy().flatten().astype(np.float32)
            except Exception as e:
                print(f"Error loading embedding {path}: {e}")

        return np.zeros(1024, dtype=np.float32)

    def load_embeddings(self):
        """Preloads all music and video embeddings/features into memory."""
        print("Preloading music embeddings and features...")
        pbar = tqdm(self.tracks, desc="Music Data")
        for track in pbar:
            # 1. Embeddings
            track.imagebind_embedding = self._read_embedding_from_disk(
                "music_embs", track.filename
            )

            # text embedding
            caption = ""
            if "rich_caption" in track.metadata:
                caption += track.metadata["rich_caption"]
            if "instrumentation" in track.metadata:
                caption += f" Instruments: {track.metadata['instrumentation']}"
            if "mood_tags" in track.metadata:
                caption += f" Mood: {', '.join(track.metadata['mood_tags'])}"
            track.caption_embedding = self._get_text_embedding(caption)

            # 2. Features
            if "mfcc_mean" in track.metadata:
                features = MusicFeatures(
                    mfcc=track.metadata.get("mfcc_mean", [0.0] * 13),
                    spectral_centroid=track.metadata.get("sc_mean", 0.0),
                    bpm=track.metadata.get("bpm", 120.0),
                    energy=track.metadata.get("rms", 0.0),
                    duration=track.metadata.get("duration", 0.0),
                )
            else:
                audio_path = os.path.join(self.data_dir, "raw_music", track.filename)
                if os.path.exists(audio_path):
                    features_dict = compute_audio_features(audio_path)
                    features = MusicFeatures(
                        mfcc=features_dict["mfcc"],
                        spectral_centroid=features_dict["spectral_centroid"],
                        energy=features_dict["energy"],
                        bpm=features_dict["bpm"],
                        duration=features_dict["duration"],
                    )
                else:
                    features = MusicFeatures(
                        mfcc=[0.0] * 13,
                        spectral_centroid=0.0,
                        energy=0.0,
                        bpm=0.0,
                        duration=0.0,
                    )

            track.features = features

        print("Preloading video embeddings...")
        pbar = tqdm(self.videos, desc="Video Data")
        for video in pbar:
            for seg in video.segments:
                # visual embedding
                seg.imagebind_embedding = self._read_embedding_from_disk(
                    "video_embs", video.filename, segment_suffix=seg.index
                )

                # caption embedding
                caption = ""
                if "visual_summary" in seg.metadata:
                    caption += seg.metadata["visual_summary"]
                if "mood_tags" in seg.metadata:
                    caption += f" Mood: {', '.join(seg.metadata['mood_tags'])}"
                seg.caption_embedding = self._get_text_embedding(caption)

    # --- Accessor Methods ---

    def get_music_track(self, index: int) -> Optional[MusicTrack]:
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return None

    def get_music_embedding(self, action_index: int) -> np.ndarray:
        track = self.get_music_track(action_index)
        if track and track.imagebind_embedding is not None:
            return track.imagebind_embedding
        return np.zeros(1024, dtype=np.float32)

    def get_music_text_embedding(self, action_index: int) -> np.ndarray:
        track = self.get_music_track(action_index)
        if track and track.caption_embedding is not None:
            return track.caption_embedding
        return np.zeros(1024, dtype=np.float32)

    def get_music_features(self, index: int) -> MusicFeatures:
        track = self.get_music_track(index)
        if track and track.features:
            return track.features
        return MusicFeatures(
            mfcc=[0.0] * 13,
            spectral_centroid=0.0,
            energy=0.0,
            bpm=0.0,
            duration=0.0,
        )

    def get_video(self, index: int) -> Optional[Video]:
        if 0 <= index < len(self.videos):
            return self.videos[index]
        return None

    # helper for filename lookups if needed, but prefer index access
    def get_video_by_filename(self, filename: str) -> Optional[Video]:
        for v in self.videos:
            if v.filename == filename:
                return v
        return None

    def get_video_embedding(self, video_idx: int, segment_index: int) -> np.ndarray:
        video = self.get_video(video_idx)
        if video and 0 <= segment_index < len(video.segments):
            emb = video.segments[segment_index].imagebind_embedding
            if emb is not None:
                return emb
        return np.zeros(1024, dtype=np.float32)

    def get_video_text_embedding(
        self, video_idx: int, segment_index: int
    ) -> np.ndarray:
        video = self.get_video(video_idx)
        if video and 0 <= segment_index < len(video.segments):
            emb = video.segments[segment_index].caption_embedding
            if emb is not None:
                return emb
        return np.zeros(1024, dtype=np.float32)
