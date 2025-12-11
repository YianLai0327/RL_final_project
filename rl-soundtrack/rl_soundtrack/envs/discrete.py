import os
from typing import Dict, Optional, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from rl_soundtrack.dataset.discrete import DiscreteDataset, MusicFeatures
from rl_soundtrack.utils.common import cosine_similarity, parse_time_str
from rl_soundtrack.utils.video_utils import render_video_soundtrack


class DiscreteEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    The goal is to select the best audio segment for a given video segment.
    """

    metadata = {"render_modes": ["human"]}

    # Class-level shared dataset (injected by train.py)
    shared_dataset: Optional[DiscreteDataset] = None

    def __init__(
        self,
        render_mode: Optional[str] = "human",
        # Params for reward weights
        w_alignment: float = 1.0,
        w_smoothness: float = 0.5,
        random: Optional[bool] = True,
        dataset: Optional[DiscreteDataset] = None,
        **kwargs,
    ):
        self.render_mode = render_mode
        super(DiscreteEnv, self).__init__()

        # Dataset Management
        # Priority: 1. Passed arg, 2. Class attribute (shared), 3. New local instance (fallback)
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = DiscreteEnv.shared_dataset

        if self.dataset is None:
            # Fallback for standalone instantiation
            print("Initializing local DiscreteDataset (no shared dataset found)...")
            self.dataset = DiscreteDataset(**kwargs)
            # For fallback, we must load embeddings explicitly if they aren't loaded
            # However, DiscreteDataset loads JSONs in init, but embeddings in load_embeddings()
            if (
                not self.dataset.tracks
                or not self.dataset.tracks[0].imagebind_embedding
            ):
                # Check if the first track has embedding loaded as a proxy
                self.dataset.load_embeddings()

        self.w_alignment = w_alignment
        self.w_smoothness = w_smoothness

        self.random = random
        self.video_idx = 0

        # Copy references for convenience
        self.tracks = self.dataset.tracks
        self.videos = self.dataset.videos
        self.n_audio = self.dataset.n_audio

        # Current Episode Data
        self.video_segments = []
        self.current_video_filename = ""
        self.n_video_segments = 0

        # Audio library aligned features are in dataset

        # Define Action Space: Select an audio index or Continue
        # 0 to n_audio-1: Select new track
        # n_audio: Continue previous track
        self.action_space = spaces.Discrete(self.n_audio + 1)

        # Define Observation Space
        # video_embedding: 1024, last_music_embedding: 1024
        self.observation_space = spaces.Dict(
            {
                "video_embedding": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
                ),
                "last_music_embedding": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
                ),
            }
        )

        self.current_step = 0
        self.last_music_embedding = np.zeros(1024, dtype=np.float32)
        self.last_audio_index = None
        self.episode_audios = []

    def _get_music_features_data(self, index: int) -> MusicFeatures:
        """
        Retrieves music features (MFCC, BPM, etc.) for the given audio index.
        Returns cached value from dataset.
        """
        dataset = cast(DiscreteDataset, self.dataset)
        return dataset.get_music_features(index)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        # Current video segment embedding
        # Access through dataset using index
        dataset = cast(DiscreteDataset, self.dataset)
        video_emb = dataset.get_video_embedding(self.video_idx, self.current_step)

        return {
            "video_embedding": video_emb,
            "last_music_embedding": self.last_music_embedding,
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.seed(seed=seed)
        self.current_step = 0
        self.last_music_embedding = np.zeros(1024, dtype=np.float32)
        self.last_audio_index = None
        self.episode_actions = []
        self.episode_audios = []
        self.episode_offsets = []
        self.current_audio_offset = 0.0

        if len(self.videos) == 0:
            print("Error: No videos found in library.")
            return self.observation_space.sample(), {}

        # Sample a video for this episode
        if self.random:
            self.video_idx = self.np_random.integers(0, len(self.videos))
        else:
            self.video_idx = (self.video_idx + 1) % len(self.videos)

        video = self.videos[self.video_idx]
        self.current_video_filename = video.filename
        self.video_segments = video.segments
        self.n_video_segments = len(self.video_segments)

        return self._get_observation(), {}

    def step(self, action: int):
        # Cast action
        if isinstance(action, np.ndarray):
            action = action.item()
        action = int(action)

        # 0. Get Segment Duration
        # seg is VideoSegment dataclass
        seg = self.video_segments[self.current_step]
        start_t = parse_time_str(seg.metadata.get("start", "00:00"))
        end_t = parse_time_str(seg.metadata.get("end", "00:00"))
        seg_duration = end_t - start_t
        if seg_duration <= 0:
            seg_duration = 5.0  # Fallback

        terminated = False
        truncated = False
        penalty = 0.0

        # Logic for determining actual audio used this step
        current_audio_idx = None
        current_offset = 0.0
        is_continue = False

        if action == self.n_audio:
            # === CONTINUE ACTION ===
            if self.last_audio_index is None:
                # Invalid continue (start of episode) -> Terminate
                current_audio_idx = 0
                current_offset = 0.0
                terminated = True
                penalty += -10.0  # Terminate penalty

            else:
                # Continue previous
                prev_idx = self.last_audio_index
                prev_features = self._get_music_features_data(prev_idx)
                total_duration = prev_features.duration

                # Check remaining time
                needed_end = self.current_audio_offset + seg_duration
                if needed_end > total_duration:
                    # Terminate!
                    terminated = True
                    penalty += -10.0  # Terminate penalty
                    current_audio_idx = prev_idx
                    current_offset = self.current_audio_offset
                else:
                    # Success continue
                    current_audio_idx = prev_idx
                    current_offset = self.current_audio_offset
                    # Update offset for NEXT step
                    self.current_audio_offset += seg_duration
                    is_continue = True

        else:
            # === NEW TRACK ACTION ===
            # specific case: Picking the SAME track as new consecutively
            if action == self.last_audio_index:
                penalty += -1.0

            # specific case: Picking the SAME track as new repeatedly
            n_repetitions = self.episode_audios.count(action)
            penalty += -0.5 * n_repetitions

            current_audio_idx = action
            current_offset = 0.0
            # Update offset for next step
            self.current_audio_offset = seg_duration

        # 1. Get Embeddings
        dataset = cast(DiscreteDataset, self.dataset)
        obs = self._get_observation()
        video_emb = obs["video_embedding"]
        video_text_emb = dataset.get_video_text_embedding(
            self.video_idx, self.current_step
        )
        music_emb = dataset.get_music_embedding(current_audio_idx)
        music_text_emb = dataset.get_music_text_embedding(current_audio_idx)

        # 2. Calculate Reward
        # 2a. Alignment
        imagebind_va_sim = cosine_similarity(video_emb, music_emb)
        caption_va_sim = cosine_similarity(video_text_emb, music_text_emb)
        alignment_score = (imagebind_va_sim + caption_va_sim) / 2.0

        # 2b. Smoothness
        smoothness_score = 0.0
        if self.last_audio_index is not None:
            # Alignment with last
            last_music_emb = dataset.get_music_embedding(self.last_audio_index)
            last_music_text_emb = dataset.get_music_text_embedding(
                self.last_audio_index
            )
            imagebind_aa_sim = cosine_similarity(music_emb, last_music_emb)
            caption_aa_sim = cosine_similarity(music_text_emb, last_music_text_emb)

            # Get features
            curr_feats = self._get_music_features_data(current_audio_idx)
            last_feats = self._get_music_features_data(self.last_audio_index)

            # BPM difference
            bpm_diff = abs(curr_feats.bpm - last_feats.bpm) / 200.0
            # Energy difference
            energy_diff = abs(curr_feats.energy - last_feats.energy)
            # Spectral Centroid difference
            sc_diff = (
                abs(curr_feats.spectral_centroid - last_feats.spectral_centroid)
                / 5000.0
            )
            # MFCC
            curr_mfcc = np.array(curr_feats.mfcc)
            last_mfcc = np.array(last_feats.mfcc)
            mfcc_dist = np.linalg.norm(curr_mfcc - last_mfcc) / 100.0

            smoothness_score = (
                imagebind_aa_sim
                + caption_aa_sim
                - bpm_diff
                - energy_diff
                - sc_diff
                - mfcc_dist
            ) / 6.0

        reward = (
            (self.w_alignment * alignment_score)
            + (self.w_smoothness * smoothness_score)
            + penalty
        )
        info = {
            "video_filename": self.current_video_filename,
            "alignment_score": alignment_score,
            "smoothness_score": smoothness_score,
            "penalty": penalty,
            "reward": reward,
        }

        # 3. Update State
        self.last_music_embedding = music_emb
        self.last_audio_index = current_audio_idx

        self.episode_actions.append(action)
        self.episode_audios.append(current_audio_idx)
        self.episode_offsets.append(current_offset)

        self.current_step += 1
        if not terminated:
            terminated = self.current_step >= self.n_video_segments

        # 4. Get next observation
        if not terminated:
            next_obs = self._get_observation()
        else:
            next_obs = obs  # Terminal observation

        return next_obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        pass

    def render_output_video(self, output_path: str):
        """
        Renders the full video with specific audio tracks selected in the episode.
        """
        if not self.current_video_filename or not self.episode_audios:
            print("No video or actions to render.")
            return

        # print(f"Rendering video to {output_path}...")

        # 1. Prepare Video Path
        # data/raw_videos/{filename}
        dataset = cast(DiscreteDataset, self.dataset)
        video_full_path = os.path.join(
            dataset.data_dir, "raw_videos", self.current_video_filename
        )

        # 2. Prepare Audio Paths
        audio_paths = []
        for action_idx in self.episode_audios:
            if 0 <= action_idx < len(self.tracks):
                filename = self.tracks[int(action_idx)].filename
                audio_path = os.path.join(dataset.data_dir, "raw_music", filename)
                audio_paths.append(audio_path)
            else:
                audio_paths.append("")

        # 3. Prepare Segment Times
        segment_times = []
        for seg in self.video_segments:
            start = parse_time_str(seg.metadata.get("start", "00:00"))
            end = parse_time_str(seg.metadata.get("end", "00:00"))
            segment_times.append((start, end))

        # 4. Render
        render_video_soundtrack(
            video_full_path,
            audio_paths,
            segment_times,
            self.episode_offsets,
            output_path,
        )
        # print(f"Successfully rendered video to {output_path}\n")

    def close(self):
        pass
