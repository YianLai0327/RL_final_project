import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rl_soundtrack.utils.audio_features import compute_audio_features
from rl_soundtrack.utils.common import cosine_similarity
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
    mood_embedding: Optional[np.ndarray] = None
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
    sigma: Optional[float] = None


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
        tracks: Optional[List[MusicTrack]] = None,
        videos: Optional[List[Video]] = None,
        video_imagebind_ref: Optional[np.ndarray] = None,
        video_caption_ref: Optional[np.ndarray] = None,
        switch_budget_model: Optional[List[Tuple[float, float, float, float]]] = None,
        **kwargs,
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
                "all-MiniLM-L6-v2",
                # "all-mpnet-base-v2",
                cache_folder=".checkpoints",
            )
        else:
            self.text_encoder = text_encoder

        # Data Storage
        if tracks is not None and videos is not None:
            self.tracks = tracks
            self.videos = videos
            self.n_audio = len(self.tracks)
        else:
            self.tracks: List[MusicTrack] = []
            self.videos: List[Video] = []

            # Initial Load of JSONs
            self._load_initial_data()
            self.n_audio = len(self.tracks)
            self._load_embeddings()

        for video in self.videos:
            if video.sigma is None:
                video.sigma = self._compute_video_sigma(video.segments)

        self.video_imagebind_ref = video_imagebind_ref
        if self.video_imagebind_ref is None:
            print("Calculating video imagebind reference embedding...")
            self.video_imagebind_ref = self._calc_video_ref_emb(
                attr="imagebind_embedding"
            )

        self.video_caption_ref = video_caption_ref
        if self.video_caption_ref is None:
            print("Calculating video caption reference embedding...")
            self.video_caption_ref = self._calc_video_ref_emb(attr="caption_embedding")

        # Model Switch Budget
        if switch_budget_model is None:
            self.switch_budget_model = self._model_switch_budget()
        else:
            self.switch_budget_model = switch_budget_model

    def split(
        self, split_ratio: float = 0.8, seed: int = 42
    ) -> Tuple["DiscreteDataset", "DiscreteDataset"]:
        """
        Splits the dataset into two instances (e.g., train/test) based on videos.
        Music tracks are shared.
        """
        # Shuffle videos
        rng = np.random.default_rng(seed)
        shuffled_videos = list(self.videos)
        rng.shuffle(shuffled_videos)

        split_idx = int(len(shuffled_videos) * split_ratio)
        train_videos = shuffled_videos[:split_idx]
        test_videos = shuffled_videos[split_idx:]

        train_ds = DiscreteDataset(
            data_dir=self.data_dir,
            text_encoder=self.text_encoder,
            tracks=self.tracks,
            videos=train_videos,
            video_imagebind_ref=self.video_imagebind_ref,
            video_caption_ref=self.video_caption_ref,
            switch_budget_model=self.switch_budget_model,
        )
        test_ds = DiscreteDataset(
            data_dir=self.data_dir,
            text_encoder=self.text_encoder,
            tracks=self.tracks,
            videos=test_videos,
            video_imagebind_ref=self.video_imagebind_ref,
            video_caption_ref=self.video_caption_ref,
            switch_budget_model=self.switch_budget_model,
        )

        print(f"Train size: {len(train_ds.videos)}")
        print(f"Test size: {len(test_ds.videos)}")

        return train_ds, test_ds

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
        emb = self.text_encoder.encode(text, convert_to_numpy=True)
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

        print(f"  [Warning] Embedding not found for {path}")
        return np.zeros(1024, dtype=np.float32)

    def _create_video_narrative(self, segment):
        """
        Strategy:
        1. Anchor: Define overall atmosphere (Mood, Energy, Pace), this is the strongest filter.
        2. Context: Insert Visual Summary (suggested to be abstracted), provide context.
        3. Target: Insert Ideal Music Description, this is usually the strongest semantic alignment point.
        """

        # 1. Extract attributes
        moods = ", ".join(segment.get("mood_tags", []))
        energy = segment.get("energy_level", "Medium")
        pace = segment.get("cut_pace", "Moderate")

        speech_status = (
            "contains dialogue" if segment.get("has_speech") else "has no speech"
        )

        # 2. Combine Narrative
        # Structure: [Mood/Energy/Pace] -> [Speech/Context] -> [Explicit Requirement]
        narrative = (
            f"The emotional tone is {moods}. "
            f"It needs {segment.get('ideal_music_description', '')}. "
            # f"A {moods} scene with {energy} energy and {pace} pace. "
            # f"The video {speech_status}. "
            # f"{segment.get('visual_summary', '')} "
        )

        return narrative

    def _create_music_narrative(self, track):
        """
        Strategy:
        1. Anchor: Define music's own atmosphere, corresponding to Video's anchor.
        2. Details: Instruments and vocals, provide specific auditory details.
        3. Description: Rich Caption provides the most natural description, placed at the end to reinforce.
        """

        # 1. Extract attributes and convert
        moods = ", ".join(track.get("mood_tags", []))
        energy = track.get("energy_level", "Medium")

        # Convert BPM to text to align with Video's "Pace"
        bpm = track.get("bpm", 120)
        if bpm < 95:
            tempo = "Slow"
        elif bpm < 130:
            tempo = "Moderate"
        else:
            tempo = "Fast"

        vocab_status = (
            "features vocals" if track.get("has_vocals") else "is instrumental"
        )
        instruments = track.get("instrumentation", "")

        # 2. Combine Narrative
        # Structure: [Mood/Energy/Tempo] -> [Vocals/Instruments] -> [Rich Caption]
        narrative = (
            f"The emotional tone is {moods}. "
            f"{track.get('rich_caption', '')}"
            # f"A {moods} music track with {energy} energy and {tempo} tempo. "
            # f"It features {instruments}. "
        )

        return narrative

    def _compute_video_sigma(self, segments: List[VideoSegment]) -> float:
        diffs: List[float] = []
        for t in range(1, len(segments)):
            v_prev = segments[t - 1].imagebind_embedding
            v_curr = segments[t].imagebind_embedding
            diffs.append(1.0 - cosine_similarity(v_curr, v_prev))
        return np.mean(diffs).item()

    def _model_switch_budget(self) -> List[Tuple[float, float, float, float]]:
        records = []
        for video in self.videos:
            T = len(video.segments)
            if T < 2:
                continue
            C = len(
                [s for s in video.segments if s.metadata.get("music_change") == True]
            )
            sigma = video.sigma
            assert sigma is not None, "Video sigma not computed"
            records.append((C, T, sigma, C / T))

        sigmas = np.array([r[2] for r in records])
        bin_edges = np.quantile(sigmas, [0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Fit a Gaussian to each bin
        bin_stats = []
        for i in range(len(bin_edges) - 1):
            rhos_in_bin = [
                float(C)
                for (C, T, sigma, rho) in records
                if bin_edges[i] <= sigma < bin_edges[i + 1]
            ]
            mu = np.mean(rhos_in_bin)
            std = np.std(rhos_in_bin) + 1e-6
            lower_sigma = bin_edges[i]
            upper_sigma = bin_edges[i + 1]
            bin_stats.append(
                (mu.item(), std.item(), lower_sigma.item(), upper_sigma.item())
            )
        # print(bin_stats)
        return bin_stats

    def sample_switch_budget(
        self, rng: np.random.Generator, video: Video, kappa: float = 1.0
    ) -> float:
        sigma = video.sigma
        assert sigma is not None, "Video sigma not computed"
        target_mu, target_std, lower_sigma, upper_sigma = self.switch_budget_model[-1]
        for mu, std, lower_sigma, upper_sigma in self.switch_budget_model:
            if lower_sigma <= sigma < upper_sigma:
                target_mu, target_std = mu, std
                break
        # We reduce the variance of the sampled switching budget to stabilize reinforcement learning while preserving dataset-level uncertainty.
        return round(target_mu + kappa * rng.normal(0, target_std))

    def _load_embeddings(self):
        """Preloads all music and video embeddings/features into memory."""
        print("Preloading music embeddings and features...")
        pbar = tqdm(self.tracks, desc="Music Data")
        for track in pbar:
            # 1. Embeddings
            track.imagebind_embedding = self._read_embedding_from_disk(
                "music_embs", track.filename
            )

            # text embedding
            caption = self._create_music_narrative(track.metadata)
            track.caption_embedding = self._get_text_embedding(caption)

            # 2. Features
            if "mfcc" in track.metadata:
                features = MusicFeatures(
                    mfcc=track.metadata.get("mfcc", [0.0] * 13),
                    spectral_centroid=track.metadata.get("spectral_centroid", 0.0),
                    bpm=track.metadata.get("bpm", 120.0),
                    energy=track.metadata.get("energy", 0.0),
                    duration=track.metadata.get("duration", 0.0),
                )
            else:
                print(f"Computing features for {track.filename}")
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
                caption = self._create_video_narrative(seg.metadata)
                seg.caption_embedding = self._get_text_embedding(caption)

    def _calc_video_ref_emb(self, attr: str = "imagebind_embedding"):
        per_video_avg_embeddings = []
        for video in self.videos:
            if not video.segments:
                continue

            video_segment_embeddings = []
            for segment in video.segments:
                if getattr(segment, attr) is None:
                    continue
                video_segment_embeddings.append(getattr(segment, attr))

            # Per-video average
            per_video_avg_embeddings.append(np.mean(video_segment_embeddings, axis=0))

        if not per_video_avg_embeddings:
            print("  [Warning] No valid video embeddings found")
            return np.zeros(1024, dtype=np.float32)

        # Across-video average
        v_ref_raw = np.mean(per_video_avg_embeddings, axis=0)

        # L2 normalization
        norm = np.linalg.norm(v_ref_raw)
        if norm > 1e-6:
            return v_ref_raw / norm
        else:
            print(
                "  [Warning] L2 norm of average video embedding is too small, setting to zero"
            )
            return np.zeros(1024, dtype=np.float32)

    # --- Accessor Methods ---

    def get_music_track(self, index: int) -> Optional[MusicTrack]:
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return None

    def get_music_embedding(self, action_index: int) -> np.ndarray:
        track = self.get_music_track(action_index)
        if track and track.imagebind_embedding is not None:
            return track.imagebind_embedding
        print(f"  [Warning] Music embedding not found for track {action_index}")
        return np.zeros(1024, dtype=np.float32)

    def get_music_text_embedding(self, action_index: int) -> np.ndarray:
        track = self.get_music_track(action_index)
        if track and track.caption_embedding is not None:
            return track.caption_embedding
        print(f"  [Warning] Music text embedding not found for track {action_index}")
        return np.zeros(1024, dtype=np.float32)

    def get_music_features(self, index: int) -> MusicFeatures:
        track = self.get_music_track(index)
        if track and track.features:
            return track.features
        print(f"  [Warning] Music features not found for track {index}")
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
        print(f"  [Warning] Video {video_idx} segment {segment_index} not found")
        return np.zeros(1024, dtype=np.float32)

    def get_video_text_embedding(
        self, video_idx: int, segment_index: int
    ) -> np.ndarray:
        video = self.get_video(video_idx)
        if video and 0 <= segment_index < len(video.segments):
            emb = video.segments[segment_index].caption_embedding
            if emb is not None:
                return emb
        print(f"  [Warning] Video {video_idx} segment {segment_index} not found")
        return np.zeros(1024, dtype=np.float32)
