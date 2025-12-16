import argparse
import math
import os
import pathlib
from pathlib import Path
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_soundtrack.dataset.discrete import DiscreteDataset
from rl_soundtrack.envs.discrete import DiscreteEnv
from rl_soundtrack.utils.common import load_config
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3 import A2C, DQN, PPO
from tqdm import tqdm

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_path, algo):
    """Loads the trained PPO model."""
    algo_class = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "MaskablePPO": MaskablePPO}[algo]
    return algo_class.load(model_path)


import time


class GreedyAgent:
    def __init__(self, env, objective: List[str] | str | None = None):
        self.env = env
        self.objective = objective

    def predict(self, obs, deterministic=True):
        rewards: np.ndarray = self.env.unwrapped.get_action_rewards(self.objective)
        reversed_rewards = rewards[::-1]
        action = int(np.argmax(reversed_rewards))
        action = self.env.unwrapped.n_audio - action
        return np.array([action]), None


class OracleConstrainedGreedyAgent:
    def __init__(self, env, C_target, objective: List[str] | str | None = None):
        self.env = env
        self.C_target = C_target
        self.objective = objective
        self.switch_steps = set()

    def get_switch_marginal_gain(self):
        rewards = self.env.unwrapped.get_action_rewards(self.objective)

        continue_action = self.env.unwrapped.n_audio
        r_continue = rewards[continue_action]

        # find best switch action
        switch_rewards = rewards[: self.env.unwrapped.n_audio]
        best_switch_action = int(np.argmax(switch_rewards))
        r_switch = switch_rewards[best_switch_action]

        delta = r_switch - r_continue
        return delta, best_switch_action

    def plan(self):
        """
        Offline planning phase: decide which steps to switch.
        """
        deltas = []

        t = 0
        done = False
        while not done:
            delta, _ = self.get_switch_marginal_gain()
            deltas.append((t, delta))

            # Take a dummy greedy action just to advance env state
            rewards = self.env.unwrapped.get_action_rewards(self.objective)
            action = int(np.argmax(rewards))
            obs, _, done, _, _ = self.env.step(action)
            t += 1

        # Pick top-C_target positive deltas
        deltas = [(t, d) for t, d in deltas if d > 0]
        deltas.sort(key=lambda x: x[1], reverse=True)

        self.switch_steps = {t for t, _ in deltas[: self.C_target]}

    def predict(self, obs, deterministic=True):
        t = self.env.unwrapped.current_step

        rewards = self.env.unwrapped.get_action_rewards(self.objective)
        continue_action = self.env.unwrapped.n_audio
        action_mask = self.env.unwrapped.action_masks()

        if not action_mask[continue_action]:
            # Forced switch:
            # pick best music, DO NOT count as budget usage, let reward function handle it
            action = int(np.argmax(rewards[:continue_action]))
        elif t in self.switch_steps:
            # Switch step: pick best music, count as budget usage
            action = int(np.argmax(rewards))
        else:
            # Continue step: do nothing
            action = continue_action

        return np.array([action]), None


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
        masks = self.env.unwrapped.action_masks()
        valid_actions = np.where(masks)[0]
        if len(valid_actions) > 0:
            # Use the environment's seeded RNG for reproducibility
            action = self.env.unwrapped.np_random.choice(valid_actions)
        else:
            # Fallback if no valid actions (shouldn't happen in this env)
            action = self.env.action_space.sample()
        return np.array([action]), None


def evaluate_episode(env, model, index, deterministic=True):
    """Runs a single episode and returns reward, steps, and actions."""
    done = False
    total_reward = 0
    steps = 0
    actions = []
    episode_history = []  # List of (start_time, music_name)
    filename = None
    rewards = {
        # "aa_continue_penalty": [],
        "illegal_penalty": [],
        "same_track_penalty": [],
        # "va_imagebind_similarity": [],
        # "va_imagebind_ref_similarity": [],
        # "va_captions_similarity": [],
        # "va_captions_ref_similarity": [],
        "vv_imagebind_similarity": [],
        # "vv_captions_similarity": [],
        # "aa_imagebind_similarity": [],
        # "aa_captions_similarity": [],
        # "aa_bpm_diff": [],
        # "aa_energy_diff": [],
        # "aa_spectral_diff": [],
        # "aa_mfcc_diff": [],
        "alignment_reward": [],
        "semantic_smoothness_reward": [],
        "acoustic_smoothness_reward": [],
        "smoothness_reward": [],
        "reward": [],
    }

    inference_times = []

    offline_time = 0
    if isinstance(model, OracleConstrainedGreedyAgent):
        env.reset(options={"video_idx": index})
        start_time = time.time()
        model.plan()
        end_time = time.time()
        offline_time = end_time - start_time
    obs, _ = env.reset(options={"video_idx": index})
    while not done:
        start_time = time.time()
        if isinstance(model, MaskablePPO):
            action_masks = get_action_masks(env)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        end_time = time.time()
        inference_times.append(end_time - start_time)

        step_action = 0
        if action.ndim == 0:
            step_action = int(action)
        elif action.ndim == 1 and action.shape[0] == 1:
            step_action = int(action[0])
        elif action.ndim == 1 and action.shape[0] == 2:
            if action[1] == 0:
                step_action = int(action[0])
            else:
                step_action = env.unwrapped.n_audio
        else:
            raise ValueError(f"Invalid action: {action}")

        actions.append(step_action)
        obs, reward, terminated, truncated, info = env.step(step_action)

        # Post-step history recording
        if step_action != env.unwrapped.n_audio:
            seg_idx = env.unwrapped.current_step - 1
            if 0 <= seg_idx < len(env.unwrapped.video_segments):
                seg = env.unwrapped.video_segments[seg_idx]
                start_time_str = seg.metadata.get("start", "00:00")

                if 0 <= step_action < len(env.unwrapped.tracks):
                    music_name = env.unwrapped.tracks[step_action].filename
                else:
                    music_name = "Unknown"  # Should not happen if logic is correct

                episode_history.append((start_time_str, music_name))

        done = terminated or truncated
        total_reward += reward
        steps += 1
        filename = info["video_filename"]
        for key, value in info.items():
            if key in rewards:
                rewards[key].append(value)

    avg_inference_time = (
        (sum(inference_times) + offline_time) / len(inference_times)
        if inference_times
        else 0.0
    )
    return (
        total_reward,
        steps,
        actions,
        rewards,
        filename,
        episode_history,
        avg_inference_time,
    )


def evaluation(
    env,
    model_dir: Path,
    model_config,
    num_episodes: int = 5,
    render=False,
    agent_type: str = "rl",
    verbosity: int = 0,
    plot_results: bool = True,
):
    """Evaluates the agent for multiple episodes."""
    if agent_type.startswith("rl/"):
        model_path = model_dir.joinpath(agent_type.split("/")[1])
        model = load_model(model_path, model_config["algo"])
        agent_name = f"RL_{model_path.stem}"
    elif agent_type == "random":
        print("Using Random Agent")
        model = RandomAgent(env)
        agent_name = "Random"
    elif agent_type.startswith("greedy"):
        # Format: greedy or greedy/objective
        parts = agent_type.split("/", 1)
        if len(parts) == 1:
            print("Using Greedy Agent (Total Reward)")
            objective = None
            agent_name = "Greedy"
        else:
            objective = parts[1].split(",")
            print(f"Using Greedy Agent with objective: {objective}")
            agent_name = f"Greedy_{objective}"
        model = GreedyAgent(env, objective=objective)
    elif agent_type.startswith("oracle_constrained_greedy/"):
        parts = agent_type.split("/", 2)
        C_target = int(parts[1])
        if len(parts) <= 2:
            objective = None
            agent_name = f"OracleConstrainedGreedy_{C_target}Switch"
            print(f"Using Oracle Constrained Greedy Agent with C_target: {C_target}")
        else:
            objective = parts[2].split(",")
            agent_name = f"OracleConstrainedGreedy_{C_target}Switch_{objective}"
            print(
                f"Using Oracle Constrained Greedy Agent with C_target: {C_target} and objective: {objective}"
            )
        model = OracleConstrainedGreedyAgent(env, C_target, objective)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    output_dir = model_dir.joinpath(f"eval_{agent_name.lower()}")
    os.makedirs(output_dir, exist_ok=True)

    continue_action = env.unwrapped.n_audio
    selected_actions = set()

    print(f"Starting evaluation for {num_episodes} episodes with {agent_name} agent...")

    all_episode_results = []
    total_inference_time = 0.0
    total_steps_all_episodes = 0

    for i in tqdm(range(num_episodes)):
        (
            total_reward,
            steps,
            actions,
            rewards,
            filename,
            episode_history,
            avg_inference_time,
        ) = evaluate_episode(env, model, i)
        selected_actions.update(set(actions))
        total_inference_time += avg_inference_time * steps
        total_steps_all_episodes += steps

        all_episode_results.append(
            {
                "filename": filename,
                "episode": i + 1,
                "reward": total_reward,
                "steps": steps,
                "actions": " ".join(f"{action:02d}" for action in actions).replace(
                    str(continue_action), "--"
                ),
                "rewards": rewards,
                "history": episode_history,
                "avg_inference_time": avg_inference_time,
            }
        )
        if render:
            save_render(env, filename, output_dir)

    if continue_action in selected_actions:
        selected_actions.remove(continue_action)
    print_evaluation(all_episode_results, selected_actions, verbosity=verbosity)

    # Calculate per-episode metrics
    rewards = []
    for res in all_episode_results:
        if res["steps"] > 0:
            # rps = res["reward"] / res["steps"]
            rewards.append(res["reward"])

    print()

    # Plotting
    if plot_results:
        plot_rewards(all_episode_results, output_dir, agent_name=agent_name)
        plot_metric_distribution(
            rewards,
            output_dir,
            agent_name=agent_name,
            metric_name="Reward",
        )

    return rewards


def print_evaluation(all_episode_results, selected_actions, verbosity=0):
    all_episode_results.sort(key=lambda x: x["filename"])
    # deduplicate (entries with the same filename -> keep first)
    all_episode_results = [
        x
        for i, x in enumerate(all_episode_results)
        if i == 0 or x["filename"] != all_episode_results[i - 1]["filename"]
    ]
    if verbosity >= 1:
        print("\n--- Evaluation Results ---")

    if verbosity >= 2:
        print("-" * 60)
        for result in all_episode_results:
            print(f"{result['filename']}")
            # Loop through history
            # Format: {switch_time}: {music_name}
            if "history" in result:
                for start_time, music_name in result["history"]:
                    print(f"  {start_time}: {music_name}")
            else:
                print("  (No history available)")
        print("-" * 60)

    rewards_per_step = []
    rewards = []
    inference_times = []

    if verbosity >= 1:
        print(
            f"{'Filename':<16} {'Reward':<8} {'Steps':<6} {'Switches':<8} {'Actions':<30}"
        )
        print("-" * 80)

    for result in all_episode_results:
        rps = result["reward"] / result["steps"] if result["steps"] > 0 else 0
        rewards_per_step.append(rps)
        rewards.append(result["reward"])
        if "avg_inference_time" in result:
            inference_times.append(result["avg_inference_time"])
        actions_str: str = result["actions"]
        switches = len(actions_str.replace(" --", "").split(" "))

        if verbosity >= 1:
            print(
                f"{result['filename'][:16]:<16} {result['reward']:<8.2f} {result['steps']:<6} {switches:<8} {actions_str:<30}"
            )

    if verbosity >= 1:
        print("-" * 80)
        print(f"Selected {len(selected_actions)} actions: {selected_actions}")
        print("-" * 80)

    if rewards:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"Reward:          {mean_reward:.2f} +/- {std_reward:.2f}")

    if rewards_per_step:
        mean_rps = np.mean(rewards_per_step)
        std_rps = np.std(rewards_per_step)
        print(f"Reward per Step: {mean_rps:.4f} +/- {std_rps:.4f}")

    if inference_times:
        mean_inf = np.mean(inference_times) * 1000
        std_inf = np.std(inference_times) * 1000
        print(f"Inference Time:  {mean_inf:.4f} +/- {std_inf:.4f} ms")
        print("-" * 80)


def plot_rewards(all_episode_results, output_dir, agent_name="agent"):
    # {model_dir}/eval_{model_stem}/{agent_name.lower()}/{title}.png
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir.joinpath("rewards.png")
    """Plots the rewards for each episode in a subplot."""

    num_episodes = len(all_episode_results)
    if num_episodes == 0:
        return

    num_rows = math.ceil(math.sqrt(num_episodes))
    # Adjust aspect ratio if too wide/tall
    num_cols = num_rows
    if num_episodes <= num_rows * (num_rows - 1):
        num_rows -= 1

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6 * num_cols, 4 * num_rows),
        sharey=True,
    )

    if num_episodes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, result in enumerate(all_episode_results):
        ax = axes[i]
        filename = result["filename"]
        truncated_filename = filename if len(filename) <= 30 else filename[:27] + "..."
        ax.set_title(f"{truncated_filename}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward Value")
        ax.set_ylim(-2, 2)

        for key, values in result["rewards"].items():
            ax.plot(values, label=key if len(key) <= 20 else key[:17] + "...")

        ax.legend(loc="lower center", ncols=2)
        ax.grid(True)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    print(f"Rewards plot saved to {save_path}")


def plot_metric_distribution(
    values, output_dir, agent_name="agent", metric_name="Metric"
):
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    title = f"dist_{metric_name.replace(' ', '_').lower()}.png"
    save_path = save_dir.joinpath(title)

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=20, edgecolor="black", alpha=0.7)
    plt.title(f"Distribution of {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    mean_val = np.mean(values)
    std_val = np.std(values)

    # Add text box with stats
    textstr = "\n".join(
        (r"$\mu={:.4f}$".format(mean_val), r"$\sigma={:.4f}$".format(std_val))
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.gca().text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"{metric_name} distribution plot saved to {save_path}")


def plot_combined_metric_distribution(
    agent_metrics: dict, output_dir, metric_name="Metric"
):
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    title = f"combined_dist_{metric_name.replace(' ', '_').lower()}.png"
    save_path = save_dir.joinpath(title)

    num_agents = len(agent_metrics)
    # Determine subplot layout
    cols = min(2, num_agents)
    rows = math.ceil(num_agents / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True
    )
    if num_agents == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Determine global min/max for consistent bins if desired, or let hist handle it
    all_values = np.concatenate(list(agent_metrics.values()))
    min_val, max_val = np.min(all_values), np.max(all_values)
    bins = np.linspace(min_val, max_val, 50)

    for i, (agent_name, values) in enumerate(agent_metrics.items()):
        ax = axes[i]
        ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(f"{agent_name}")
        if i >= (rows - 1) * cols:
            ax.set_xlabel(metric_name)
        if i % cols == 0:
            ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        mean_val = np.mean(values)
        std_val = np.std(values)
        textstr = "\n".join(
            (r"$\mu={:.4f}$".format(mean_val), r"$\sigma={:.4f}$".format(std_val))
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.95,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    print(f"Combined {metric_name} distribution plot saved to {save_path}")

    # --- CDF Plot ---
    title_cdf = f"combined_cdf_{metric_name.replace(' ', '_').lower()}.png"
    save_path_cdf = save_dir.joinpath(title_cdf)

    fig_cdf, ax_cdf = plt.subplots(1, 1, figsize=(5, 4))
    for agent_name, values in agent_metrics.items():
        sorted_vals = np.sort(values)
        y_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(y_vals, sorted_vals, marker=".", linestyle="-", label=agent_name)

    ax_cdf.set_title(f"CDF of {metric_name}")
    ax_cdf.set_xlabel("Cumulative Probability")
    ax_cdf.set_ylabel(metric_name)
    ax_cdf.grid(True, alpha=0.3)
    # ax_cdf.legend()
    fig_cdf.tight_layout()
    fig_cdf.savefig(save_path_cdf, dpi=300)
    print(f"Combined {metric_name} CDF plot saved to {save_path_cdf}")


def save_render(env, filename, output_dir):
    """Saves the rendered output if supported and requested."""
    render_dir = output_dir.joinpath("render")
    os.makedirs(render_dir, exist_ok=True)
    final_output_path = render_dir.joinpath(filename)

    if hasattr(env.unwrapped, "render_output_video"):
        # print(f"Saving render to {final_output_path}...")
        env.unwrapped.render_output_video(final_output_path)
    else:
        print("Environment does not support render_output_video.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL agent for Soundtrack")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "-n", "--episode", type=int, default=5, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "-s", "--set", type=str, default="test", help="Dataset set to evaluate"
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=str,
        default="data/medium",
        help="Path to dataset dir",
    )
    parser.add_argument(
        "-r", "--render", action="store_true", help="Render environment"
    )
    parser.add_argument(
        "-a",
        "--agent",
        action="append",
        type=str,
        help="Agent type: rl/{ckpt_name} (default), random, greedy, greedy/{objective}. Can be specified multiple times.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity level: 0=Metrics, 1=Table",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create environment
    print(
        "Preloading shared DiscreteDataset to avoid redundant loading in subprocesses..."
    )
    dataset = DiscreteDataset(data_dir=str(Path.cwd().joinpath(args.dataset_dir)))
    train_ds, test_ds = dataset.split(
        split_ratio=config["train"]["split_ratio"], seed=config["train"]["seed"]
    )

    # Create environment
    env = gym.make(
        config["env"]["id"],
        render_mode="human",
        random=False,
        dataset=train_ds if args.set == "train" else test_ds,
        **config["env"]["kwargs"],
    )
    env.reset(seed=config["train"]["seed"])

    # Evaluate

    model_dir = Path.cwd().joinpath(args.model_dir)

    # Handle defaults
    if not args.agent:
        agent_types = ["rl"]
    else:
        agent_types = args.agent

    # If multiple agents, we skip individual plots and do a combined plot at the end
    plot_individual = len(agent_types) == 1
    all_agents_rewards = {}

    for agent_type in agent_types:
        # Determine agent name for dict key (approximated logic from evaluation function)
        if agent_type.startswith("rl"):
            if "/" in agent_type:
                name_suffix = agent_type.split("/")[1]
                current_agent_name = f"RL_{name_suffix}"
            else:
                current_agent_name = "RL"
        elif agent_type == "random":
            current_agent_name = "Random"
        elif agent_type.startswith("greedy"):
            if "/" in agent_type:
                current_agent_name = f"Greedy_{agent_type.split('/')[1]}"
            else:
                current_agent_name = "Greedy"
        elif agent_type.startswith("oracle_constrained_greedy"):
            if "/" in agent_type:
                current_agent_name = (
                    f"OracleConstrainedGreedy_{agent_type.split('/')[1]}"
                )
            else:
                current_agent_name = "OracleConstrainedGreedy"
        else:
            current_agent_name = agent_type

        rewards_per_step = evaluation(
            env,
            model_dir,
            config["agent"],
            num_episodes=args.episode,
            render=args.render,
            agent_type=agent_type,
            verbosity=args.verbosity,
            plot_results=plot_individual,
        )
        all_agents_rewards[current_agent_name] = rewards_per_step

    if not plot_individual:
        # Combined plot
        # Save to model_dir/eval_comparison
        output_dir = model_dir.joinpath("eval_comparison")
        plot_combined_metric_distribution(
            all_agents_rewards, output_dir, metric_name="Reward"
        )


if __name__ == "__main__":
    main()
