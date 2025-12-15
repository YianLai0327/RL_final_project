#!/usr/bin/env python3
"""
Interactive script to manually select actions for a video and check the reward.
This allows you to explore different soundtrack choices and see their rewards.
"""

import argparse
import os
import sys
sys.path.append('../rl_soundtrack')
from pathlib import Path
from typing import Optional, cast
import json

import numpy as np
from rl_soundtrack.dataset.discrete import DiscreteDataset
from rl_soundtrack.envs.discrete import DiscreteEnv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ManualActionExplorer:
    def __init__(self, dataset: DiscreteDataset, env: DiscreteEnv):
        self.dataset = dataset
        self.env = env
        self.episode_steps = []
        self.total_reward = 0.0

    def print_header(self):
        print("\n" + "=" * 80)
        print("MANUAL ACTION EXPLORER - Select Soundtrack for Video")
        print("=" * 80)

    def print_video_info(self):
        video = self.env.videos[self.env.video_idx]
        print(f"\n📹 Video: {video.filename}")
        print(f"   Total Segments: {len(self.env.video_segments)}")
        print(f"   Switch Budget: {self.env.C_target} switches allowed")

    def print_current_step(self, step_idx: int):
        if step_idx >= len(self.env.video_segments):
            print(f"\n❌ Episode finished! No more segments.")
            return False

        seg = self.env.video_segments[step_idx]
        start = seg.metadata.get("start", "unknown")
        end = seg.metadata.get("end", "unknown")
        print(f"\n[Step {step_idx + 1}/{len(self.env.video_segments)}]")
        print(f"   Segment: {start} -> {end}")
        print(f"   Caption: {seg.metadata.get('description', 'N/A')}")
        return True
    
    def print_all_time_segments(self, verbose: bool = False):
        
        if verbose:
            print("\n⏱️ Video Time Segments:")
            print(f"{'Segment #':<10} {'Start':<10} {'End':<10} {'Caption'}")
            print("-" * 60)
        start_time = []
        for idx, seg in enumerate(self.env.video_segments):
            start = seg.metadata.get("start", "unknown")
            end = seg.metadata.get("end", "unknown")
            caption = seg.metadata.get("description", "N/A")
            if verbose:
                print(f"{idx + 1:<10} {start:<10} {end:<10} {caption}")
            start_time.append(self.get_sec(start))
        return start_time

    def get_sec(self, time_str):
        """Get seconds from time."""
        m, s = time_str.split(':')
        return int(m) * 60 + int(s)
    
    def map_to_nearest_ts(self, t, ts):
        result = [0] * len(ts)

        for x in t:
            # find index of nearest ts value
            nearest_idx = min(
                range(len(ts)),
                key=lambda i: abs(ts[i] - x)
            )
            result[nearest_idx] = 1

        return result

    def print_action_rewards(self):
        """Calculate and display rewards for all possible actions."""
        rewards = self.env.unwrapped.get_action_rewards()
        
        print(f"\n📊 Action Rewards:")
        print(f"{'Music ID':<12} {'Track Name':<40} {'Reward':<10}")
        print("-" * 62)

        # Show music options
        for idx in range(min(5, self.env.n_audio)):  # Show top 10 or less
            music = self.env.tracks[idx]
            reward = rewards[idx]
            music_name = Path(music.filename).stem[:38]
            print(f"{idx:<12} {music_name:<40} {reward:>9.4f}")

        if self.env.n_audio > 10:
            print(f"... and {self.env.n_audio - 10} more music tracks")

        # Show continue option
        continue_idx = self.env.n_audio
        continue_reward = rewards[continue_idx]
        continue_status = "✓ Available" if continue_reward > -np.inf else "✗ Not available"
        print("-" * 62)
        print(f"{continue_idx:<12} {'[CONTINUE]':<40} {continue_reward:>9.4f} {continue_status}")

    def print_detailed_reward_breakdown(self, action: int, info: dict):
        """Print detailed breakdown of reward components for an action."""
        # Take the action and get full info
        saved_state = {
            "current_step": self.env.current_step,
            "last_audio_embedding": self.env.last_audio_embedding.copy(),
            "last_audio_index": self.env.last_audio_index,
            "last_audio_ended": self.env.last_audio_ended,
            "episode_actions": list(self.env.episode_actions),
            "episode_audios": list(self.env.episode_audios),
            "episode_offsets": list(self.env.episode_offsets),
            "current_audio_offset": self.env.current_audio_offset,
            "switch_count": self.env.switch_count,
            "audio_emb_sum": self.env.audio_emb_sum.copy(),
            "audio_step_count": self.env.audio_step_count,
        }

        # obs, reward, terminated, truncated, info = self.env.step(action)

        # Restore state
        self.env.current_step = saved_state["current_step"]
        self.env.last_audio_embedding = saved_state["last_audio_embedding"]
        self.env.last_audio_index = saved_state["last_audio_index"]
        self.env.last_audio_ended = saved_state["last_audio_ended"]
        self.env.episode_actions = saved_state["episode_actions"]
        self.env.episode_audios = saved_state["episode_audios"]
        self.env.episode_offsets = saved_state["episode_offsets"]
        self.env.current_audio_offset = saved_state["current_audio_offset"]
        self.env.switch_count = saved_state["switch_count"]
        self.env.audio_emb_sum = saved_state["audio_emb_sum"]
        self.env.audio_step_count = saved_state["audio_step_count"]

        # Print breakdown
        print(f"\n🎵 Reward Breakdown for Action {action}:")
        print("-" * 50)

        # Key reward components
        components = {
            "alignment_reward": "Video-Music Alignment",
            "semantic_smoothness_reward": "Semantic Smoothness",
            "acoustic_smoothness_reward": "Acoustic Smoothness",
            "theme_reward": "Thematic Coherence",
            "switch_reward": "Switch Penalty",
        }

        for key, label in components.items():
            if key in info:
                print(f"  {label:<25}: {info[key]:>9.4f}")

        print("-" * 50)
        print(f"  {'Total Reward':<25}: {info['reward']:>9.4f}")

    def _get_user_action(self) -> Optional[int]:
        """Get action input from user with validation."""
        action_masks = self.env.unwrapped.action_masks()

        while True:
            try:
                user_input = input("\n➤ Enter action (music ID or 'c' for continue, 'q' to quit): ").strip().lower()

                if user_input == 'q':
                    return None

                if user_input == 'c':
                    action = self.env.n_audio
                    if not action_masks[action]:
                        print("   ❌ Cannot continue now!")
                        continue
                    return action

                action = int(user_input)
                if action < 0 or action > self.env.n_audio:
                    print(f"   ❌ Invalid action. Must be 0-{self.env.n_audio} or 'c'")
                    continue

                if not action_masks[action]:
                    print(f"   ❌ Action {action} is not available!")
                    continue

                return action

            except ValueError:
                print("   ❌ Invalid input. Please enter a number or 'c'")

    def get_user_action(self, switch_seq, current_step, music_id, music_id_list) -> Optional[int]:
        """Get action input from user with validation."""
        action_masks = self.env.unwrapped.action_masks()
        print(switch_seq, current_step)
        print(f'musicid: {music_id}')
        while True:
            try:
                # user_input = input("\n➤ Enter action (music ID or 'c' for continue, 'q' to quit): ").strip().lower()
                if switch_seq[current_step]==1:
                    user_input = str(music_id_list[music_id])
                else:
                    user_input = 'c'
                    action = self.env.n_audio
                    if not action_masks[action]:
                        print("   ❌ Cannot continue now!")
                        if music_id + 1 == len(music_id_list):
                            music_id = -1
                        
                        return music_id_list[music_id], music_id + 1
                    return action, music_id

                action = int(user_input)
                

                if not action_masks[action]:
                    print(f"   ❌ Action {action} is not available!")
                    continue
                if music_id + 1 == len(music_id_list):
                    music_id = -1
                return action, music_id + 1

            except ValueError:
                print("   ❌ Invalid input. Please enter a number or 'c'")

    def split_data(self):
        video = self.env.videos[self.env.video_idx]
        with open('split.json', 'r') as fp:
            data_dict = json.load(fp)
        print(f"\n📹 Video: {video.filename}")
        time_seq = data_dict[os.path.splitext(video.filename)[0] + '.mp3']
        time_seq.insert(0, 0)
        time_seq.remove(max(time_seq))
        print(f"split data: {time_seq}")
        return time_seq
        # print(f"   Total Segments: {len(self.env.video_segments)}")
        # print(f"   Switch Budget: {self.env.C_target} switches allowed")

    def audio_idx_calculate(self, audio_idx):
        audio_video_map = [0, 7, 20, 25, 31, 38, 50, 61]
        return list(range(audio_video_map[audio_idx], audio_video_map[audio_idx + 1]))

    def run_episode(self, video_idx: Optional[int] = None, audio_idx: Optional[int] = None):
        """Run an interactive episode where user selects actions."""
        self.total_reward = 0
        self.print_header()

        # Reset environment
        options = {}
        if video_idx is not None:
            options["video_idx"] = video_idx
        obs, _ = self.env.reset(options=options)

        self.print_video_info()

        timesplit = self.split_data()
        timestamp = self.print_all_time_segments()
        switch_seq = self.map_to_nearest_ts(timesplit, timestamp)
        music_id_list = self.audio_idx_calculate(audio_idx)
        music_id = 0
        
        # Iterate through segments
        while self.env.current_step < len(self.env.video_segments):
            if not self.print_current_step(self.env.current_step):
                break
            
            # Show available actions and their rewards
            # self.print_action_rewards()
            
            
            # Get user action
            action, music_id = self.get_user_action(switch_seq, self.env.current_step, music_id, music_id_list)
            if action is None:
                print("\n✗ Episode cancelled by user.")
                return
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.episode_steps.append((action, reward, info))
            self.total_reward += reward
            
            # Show reward breakdown
            # self.print_detailed_reward_breakdown(action, info)
            
            # Show action summary
            if action == self.env.n_audio:
                print(f"✓ Continued previous track. Reward: {reward:.4f}")
            else:
                music = self.env.tracks[action]
                print(f"✓ Selected '{Path(music.filename).stem}'. Reward: {reward:.4f}")

            if terminated:
                print(f"\n⚠️ Episode terminated early!")
                break


        # Print episode summary
        # self.print_episode_summary()
        return self.total_reward

    def print_episode_summary(self):
        """Print summary of the episode."""
        print("\n" + "=" * 80)
        print("EPISODE SUMMARY")
        print("=" * 80)
        print(f"Total Steps: {len(self.episode_steps)}")
        print(f"Total Reward: {self.total_reward:.4f}")
        print(f"Average Reward per Step: {self.total_reward / max(1, len(self.episode_steps)):.4f}")
        print(f"Switches Used: {self.env.switch_count}/{self.env.C_target}")

        if self.episode_steps:
            print(f"\n📝 Action Sequence:")
            for i, (action, reward, _) in enumerate(self.episode_steps):
                if action == self.env.n_audio:
                    action_str = "[CONTINUE]"
                else:
                    action_str = f"Music {action}"
                print(f"   Step {i + 1}: {action_str:<20} Reward: {reward:>9.4f}")



def visualize_matrix(mat):
    mat = np.asarray(mat)
    n = mat.shape[0]

    # softmax along entire matrix (for color intensity)
    # exp_mat = np.exp(mat - mat.max())
    # mat_soft = exp_mat / exp_mat.sum()

    # y 軸 (audio axis) softmax
    # exp_mat = np.exp(mat - mat.max(axis=1, keepdims=True))
    # mat_soft = exp_mat / exp_mat.sum(axis=1, keepdims=True)


    cmap = LinearSegmentedColormap.from_list("white_green", ["white", "green"])

    plt.figure()

    im = plt.imshow(mat, cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label("reward (Normalization)")

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{mat[i, j]:.2f}",
                     ha="center", va="center", color="black")

    plt.title("Confusion Matrix (Normalization along video axis)")
    plt.xlabel("Audio index")
    plt.ylabel("Video index")

    plt.xticks(range(n))
    plt.yticks(range(n))

    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Manually select actions for a video and check rewards"
    )
    parser.add_argument(
        "--video-idx",
        type=int,
        default=None,
        help="Specific video index to use (default: random)",
    )
    parser.add_argument(
        "--w-alignment",
        type=float,
        default=1.0,
        help="Weight for alignment reward",
    )
    parser.add_argument(
        "--w-smoothness",
        type=float,
        default=0.5,
        help="Weight for smoothness reward",
    )
    parser.add_argument(
        "--w-switch",
        type=float,
        default=0.3,
        help="Weight for switch penalty",
    )
    parser.add_argument(
        "--w-theme",
        type=float,
        default=0.3,
        help="Weight for theme reward",
    )

    args = parser.parse_args()

    print("Loading dataset and environment...")

    
    dataset = DiscreteDataset(
        data_dir=str(Path.cwd().joinpath('data'))
    )
    print(f"✓ Loaded {len(dataset.tracks)} music tracks")
    print(f"✓ Loaded {len(dataset.videos)} videos")

    # Initialize environment
    env = DiscreteEnv(
        dataset=dataset,
        w_alignment=args.w_alignment,
        w_smoothness=args.w_smoothness,
        w_switch=args.w_switch,
        w_theme=args.w_theme,
        random=False,
    )

    # Create explorer
    explorer = ManualActionExplorer(dataset, env)

    # Run interactive episode
    confusion_matrix = np.zeros((7, 7))
    for idx in range(7):
        for jdx in range(7):
            confusion_matrix[idx][jdx] = explorer.run_episode(video_idx=idx, audio_idx=jdx)
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    print(f"\nConfusion Matrix:\n{confusion_matrix}")
    visualize_matrix(confusion_matrix)



if __name__ == "__main__":
    main()
