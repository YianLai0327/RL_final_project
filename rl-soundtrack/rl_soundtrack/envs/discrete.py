import os
from typing import Dict, List, Optional, cast

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
        w_switch: float = 0.3,
        w_theme: float = 0.3,
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

        self.w_alignment = w_alignment
        self.w_smoothness = w_smoothness
        self.w_switch = w_switch
        self.w_theme = w_theme

        self.random = random
        self.video_idx = 0

        # Copy references for convenience
        self.tracks = self.dataset.tracks
        self.videos = self.dataset.videos
        self.n_audio = self.dataset.n_audio

        # Current Episode Data
        self.video_segments = []
        self.current_video_filename = ""

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
                "last_audio_embedding": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
                ),
                "switch_ratio": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.current_step = 0
        self.last_audio_embedding = np.zeros(1024, dtype=np.float32)
        self.last_audio_index = None
        self.episode_audios = []

        self.video_imagebind_ref_emb = self.dataset.video_imagebind_ref
        self.video_caption_ref_emb = self.dataset.video_caption_ref

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
            "last_audio_embedding": self.last_audio_embedding,
            "switch_ratio": np.array(
                [self.switch_count / max(1, len(self.video_segments))], dtype=np.float32
            ),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_audio_embedding = np.zeros(1024, dtype=np.float32)
        self.last_audio_index = None
        self.last_audio_ended = False
        # Rendering
        self.episode_actions = []  # List of actions taken in this episode
        self.episode_audios = []  # List of audio indices used in this episode
        self.episode_offsets = []  # List of audio offsets used in this episode
        self.current_audio_offset = 0.0
        # Switching Budget
        self.switch_count = 0
        # Thematic Coherence
        self.audio_emb_sum = np.zeros(1024, dtype=np.float32)
        self.audio_step_count = 0

        if len(self.videos) == 0:
            print("Error: No videos found in library.")
            return self.observation_space.sample(), {}

        # Sample a video for this episode
        if self.random:
            self.video_idx = int(self.np_random.integers(0, len(self.videos)))
        elif options is not None and "video_idx" in options:
            self.video_idx = options["video_idx"] % len(self.videos)
        else:
            self.video_idx = (self.video_idx + 1) % len(self.videos)

        video = self.videos[self.video_idx]
        self.current_video_filename = video.filename
        self.video_segments = video.segments
        self.C_target = cast(DiscreteDataset, self.dataset).sample_switch_budget(
            self.np_random, video
        )

        return self._get_observation(), {}

    def action_masks(self) -> list[bool]:
        """Returns a boolean mask of valid actions for Maskable Policies"""
        mask = [True] * (self.n_audio + 1)

        # First step, no previous audio to continue
        if self.last_audio_index is None:
            mask[self.n_audio] = False

        # Last music ended, no more music to continue
        if self.last_audio_ended:
            mask[self.n_audio] = False

        return mask

    def step(self, action: int):
        if action == self.n_audio:
            continue_flag = 1
            music_idx = 0  # arbitrary index is fine
        else:
            continue_flag = 0
            music_idx = action

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

        # Rewards
        reward = 0
        ## special penalties
        illegal_penalty = 0
        same_track_penalty = 0
        aa_continue_penalty = 0
        ## video-music rewards
        va_imagebind_similarity = 1
        va_imagebind_ref_similarity = 1
        va_captions_similarity = 1
        va_captions_ref_similarity = 1
        alignment_reward = 1
        ## music-music rewards/penalties
        vv_imagebind_similarity = 1
        vv_captions_similarity = 1
        aa_imagebind_similarity = 1
        aa_captions_similarity = 1
        aa_bpm_diff = 0
        aa_energy_diff = 0
        aa_spectral_diff = 0
        aa_mfcc_diff = 0
        semantic_smoothness_reward = 1
        acoustic_smoothness_reward = 1
        smoothness_gating_factor = 1
        smoothness_reward = 1
        ## global rewards
        theme_reward = 0
        switch_reward = 0

        # Logic for determining actual audio used this step
        current_audio_idx = None
        current_offset = 0.0

        # |           | First segment | Other segments |
        # | --------- | ------------- | -------------- |
        # | Continue  |       illegal |        special |
        # | New music |        normal |        special |
        if continue_flag == 1:
            # === CONTINUE ACTION ===
            if self.last_audio_index is None:
                # Invalid continue (start of episode) -> Terminate
                illegal_penalty += -5.0  # Terminate penalty
                terminated = True
                current_audio_idx = 0
                print("Maskable won't continue at start of episode")
            else:
                # Continue previous
                prev_idx = self.last_audio_index

                if self.last_audio_ended:
                    # Invalid continue (end of music) -> Terminate
                    illegal_penalty += -5.0  # Terminate penalty
                    terminated = True
                    current_audio_idx = 0
                    print("Maskable won't continue at end of music")
                else:
                    # Success continue
                    current_audio_idx = prev_idx
                    current_offset = self.current_audio_offset
                    self.current_audio_offset += seg_duration
                    # Update end flag
                    audio_duration = self._get_music_features_data(prev_idx).duration
                    self.last_audio_ended = self.current_audio_offset >= audio_duration
        else:
            # === NEW TRACK ACTION ===
            # specific case: Picking the SAME track as new consecutively
            if music_idx == self.last_audio_index:
                same_track_penalty += -1.0

            # specific case: Picking the SAME track as new repeatedly
            n_repetitions = self.episode_actions.count(music_idx)
            same_track_penalty += -1.0 * n_repetitions

            # Update audio index and offset
            current_audio_idx = music_idx
            current_offset = 0.0
            self.current_audio_offset = seg_duration
            # Update end flag
            audio_duration = self._get_music_features_data(current_audio_idx).duration
            self.last_audio_ended = self.current_audio_offset >= audio_duration
        reward += illegal_penalty
        reward += same_track_penalty

        if continue_flag == 0 and self.last_audio_index is not None:
            self.switch_count += 1

        # 1. Get State
        dataset = cast(DiscreteDataset, self.dataset)
        obs = self._get_observation()

        # 2. Calculate Reward
        # 2a. Video-Music Alignment
        # Get embeddings
        video_emb = obs["video_embedding"]
        video_text_emb = dataset.get_video_text_embedding(
            self.video_idx, self.current_step
        )
        music_emb = dataset.get_music_embedding(current_audio_idx)
        music_text_emb = dataset.get_music_text_embedding(current_audio_idx)
        # Calculate similarity
        va_imagebind_similarity = cosine_similarity(video_emb, music_emb)
        va_captions_similarity = cosine_similarity(video_text_emb, music_text_emb)
        va_imagebind_ref_similarity = cosine_similarity(
            self.video_imagebind_ref_emb, music_emb
        )
        va_captions_ref_similarity = cosine_similarity(
            self.video_caption_ref_emb, music_text_emb
        )
        # Calculate reward
        alignment_gating_factor = np.exp(
            -0.01 * max(0.0, self.current_audio_offset - seg_duration)
        )
        alignment_reward = (
            # alignment_gating_factor
            1
            * (
                (va_imagebind_similarity - va_imagebind_ref_similarity + 0.2) * 2.5
                + (va_captions_similarity - va_captions_ref_similarity + 0.3) * 2.5
            )
            / 2.0
        )  # ~ [0,1]
        reward += self.w_alignment * alignment_reward

        # 2b. Music-Music Smoothness
        if self.last_audio_index is not None:
            # Get embeddings
            last_music_emb = obs["last_audio_embedding"]
            last_music_text_emb = dataset.get_music_text_embedding(
                self.last_audio_index
            )
            last_video_emb = dataset.get_video_embedding(
                self.video_idx, self.current_step - 1
            )
            last_video_text_emb = dataset.get_video_text_embedding(
                self.video_idx, self.current_step - 1
            )
            # Calculate similarity
            aa_imagebind_similarity = cosine_similarity(music_emb, last_music_emb)
            aa_captions_similarity = cosine_similarity(
                music_text_emb, last_music_text_emb
            )
            vv_imagebind_similarity = cosine_similarity(video_emb, last_video_emb)
            vv_captions_similarity = cosine_similarity(
                video_text_emb, last_video_text_emb
            )

            # Get features
            curr_feats = self._get_music_features_data(current_audio_idx)
            last_feats = self._get_music_features_data(self.last_audio_index)
            # BPM difference
            aa_bpm_diff = -abs(curr_feats.bpm - last_feats.bpm) / 100.0  # [-1,0]
            # Energy difference
            aa_energy_diff = -abs(curr_feats.energy - last_feats.energy)  # [-1,0]
            # Spectral Centroid difference
            aa_spectral_diff = (
                -abs(curr_feats.spectral_centroid - last_feats.spectral_centroid)
                / 2000.0
            )  # [-1,0]
            # MFCC
            curr_mfcc = np.array(curr_feats.mfcc)
            last_mfcc = np.array(last_feats.mfcc)
            # aa_mfcc_diff = -np.linalg.norm(curr_mfcc - last_mfcc) / 300.0
            aa_mfcc_diff = (cosine_similarity(curr_mfcc, last_mfcc) - 1) / 2.0  # [-1,0]

            # a. 1 - (vv_difference) = 1 - (1 - vv_similarity)
            # smoothness_gating_factor = np.clip(vv_imagebind_similarity, 0, 1)
            # b. exp(-k * vv_difference) = exp(-k * (1 - vv_similarity))
            smoothness_gating_factor = np.exp(-5 * (1 - vv_imagebind_similarity))
            semantic_smoothness_reward = (aa_imagebind_similarity + 1) / 2.0  # [0,1]
            acoustic_smoothness_reward = (
                aa_bpm_diff + aa_spectral_diff + aa_mfcc_diff
            ) / 3.0 + 1  # [0,1]
            smoothness_reward = smoothness_gating_factor * (
                0.5 * semantic_smoothness_reward + 0.5 * acoustic_smoothness_reward
            )
            reward += self.w_smoothness * smoothness_reward

        self.audio_emb_sum += music_emb
        self.audio_step_count += 1

        # 3. Update State
        self.last_audio_embedding = music_emb
        self.last_audio_index = current_audio_idx

        self.episode_actions.append(action)
        self.episode_audios.append(current_audio_idx)
        self.episode_offsets.append(current_offset)

        self.current_step += 1
        if not terminated:
            terminated = self.current_step >= len(self.video_segments)

        if terminated:
            # Switching Budget
            switch_reward = -self.w_switch * abs(self.switch_count - self.C_target)
            reward += switch_reward
            # Thematic Coherence
            z = self.audio_emb_sum / self.audio_step_count
            z = z / (np.linalg.norm(z) + 1e-6)

            theme_reward = 0.0
            for idx in self.episode_audios:
                emb = dataset.get_music_embedding(idx)
                theme_reward += cosine_similarity(emb, z)
            theme_reward = self.w_theme * (theme_reward / len(self.episode_audios))
            reward += theme_reward

        # 4. Get next observation
        if not terminated:
            next_obs = self._get_observation()
        else:
            next_obs = obs  # Terminal observation

        info = {
            "video_filename": self.current_video_filename,
            "aa_continue_penalty": aa_continue_penalty,
            "illegal_penalty": illegal_penalty,
            "same_track_penalty": same_track_penalty,
            "va_imagebind_similarity": va_imagebind_similarity,
            "va_imagebind_ref_similarity": va_imagebind_ref_similarity,
            "va_captions_similarity": va_captions_similarity,
            "va_captions_ref_similarity": va_captions_ref_similarity,
            "vv_imagebind_similarity": vv_imagebind_similarity,
            "vv_captions_similarity": vv_captions_similarity,
            "aa_imagebind_similarity": aa_imagebind_similarity,
            "aa_captions_similarity": aa_captions_similarity,
            "aa_bpm_diff": aa_bpm_diff,
            "aa_energy_diff": aa_energy_diff,
            "aa_spectral_diff": aa_spectral_diff,
            "aa_mfcc_diff": aa_mfcc_diff,
            "smoothness_gating_factor": smoothness_gating_factor,
            "alignment_reward": alignment_reward,
            "semantic_smoothness_reward": semantic_smoothness_reward,
            "acoustic_smoothness_reward": acoustic_smoothness_reward,
            "smoothness_reward": smoothness_reward,
            "switch_reward": switch_reward,
            "theme_reward": theme_reward,
            "reward": reward,
        }

        return next_obs, float(reward), terminated, truncated, info

    def get_action_rewards(
        self, key_from_info: str | List[str] | None = None
    ) -> np.ndarray:
        """
        Calculates the reward for all possible actions without modifying the state.
        Returns:
            np.ndarray: Array of rewards for each action.
        """
        # Save state
        saved_state = {
            "current_step": self.current_step,
            "last_audio_embedding": self.last_audio_embedding.copy(),
            "last_audio_index": self.last_audio_index,
            "last_audio_ended": self.last_audio_ended,
            "episode_actions": list(self.episode_actions),
            "episode_audios": list(self.episode_audios),
            "episode_offsets": list(self.episode_offsets),
            "current_audio_offset": self.current_audio_offset,
            "switch_count": self.switch_count,
            "audio_emb_sum": self.audio_emb_sum.copy(),
            "audio_step_count": self.audio_step_count,
        }

        n_action = self.n_audio + 1
        rewards = np.zeros(n_action, dtype=np.float32)

        # Iterate over all actions
        for action in range(n_action):
            # Check if action is valid
            masks = self.action_masks()
            if not masks[action]:
                rewards[action] = -np.inf
                continue

            # Step
            _, reward, _, _, info = self.step(action)
            rewards[action] = reward
            if key_from_info is not None:
                if isinstance(key_from_info, list):
                    for key in key_from_info:
                        rewards[action] += info[key]
                else:
                    rewards[action] = info[key_from_info]

            # Restore state
            self.current_step = saved_state["current_step"]
            self.last_audio_embedding = saved_state["last_audio_embedding"].copy()
            self.last_audio_index = saved_state["last_audio_index"]
            self.last_audio_ended = saved_state["last_audio_ended"]
            self.episode_actions = list(saved_state["episode_actions"])
            self.episode_audios = list(saved_state["episode_audios"])
            self.episode_offsets = list(saved_state["episode_offsets"])
            self.current_audio_offset = saved_state["current_audio_offset"]
            self.switch_count = saved_state["switch_count"]
            self.audio_emb_sum = saved_state["audio_emb_sum"].copy()
            self.audio_step_count = saved_state["audio_step_count"]

        return rewards

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
        # data/video_library/{filename}
        dataset = cast(DiscreteDataset, self.dataset)
        video_full_path = os.path.join(
            dataset.data_dir, "video_library", self.current_video_filename
        )

        # 2. Prepare Audio Paths
        audio_paths = []
        for action_idx in self.episode_audios:
            if 0 <= action_idx < len(self.tracks):
                filename = self.tracks[int(action_idx)].filename
                audio_path = os.path.join(dataset.data_dir, "music_library", filename)
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
