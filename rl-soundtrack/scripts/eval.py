import argparse
import os
import pathlib
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register
from rl_soundtrack.dataset.discrete import DiscreteDataset
from rl_soundtrack.envs.discrete import DiscreteEnv
from rl_soundtrack.utils.common import load_config
from stable_baselines3 import PPO
from tqdm import tqdm

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_path):
    """Loads the trained PPO model."""
    return PPO.load(
        model_path,
    )


def evaluate_episode(env, model, seed, deterministic=True):
    """Runs a single episode and returns reward, steps, and actions."""
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0
    steps = 0
    actions = []
    info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        if len(action) == 1:
            actions.append(action.item())
        elif len(action) == 2:
            if action[1] == 0:
                actions.append(action[0])
            else:
                actions.append(env.unwrapped.n_audio)
        else:
            raise ValueError(f"Invalid action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    return total_reward, steps, actions, info


def evaluation(env, model_path, num_episodes=5, render=False):
    """Evaluates the agent for multiple episodes."""
    model = load_model(model_path)
    continue_action = env.unwrapped.n_audio
    selected_actions = set()

    print(f"Starting evaluation for {num_episodes} episodes...")

    all_episode_results = []
    for i in tqdm(range(num_episodes)):
        total_reward, steps, actions, info = evaluate_episode(env, model, i + 42)
        selected_actions.update(set(actions))
        all_episode_results.append(
            {
                "episode": i + 1,
                "reward": total_reward,
                "steps": steps,
                "actions": " ".join(f"{action:02d}" for action in actions).replace(
                    str(continue_action), "--"
                ),
            }
        )
        if render:
            save_render(env, info["video_filename"], model_path)

    print("\n--- Evaluation Results ---")
    print(f"{'Episode':<8} {'Reward':<8} {'Steps':<8} {'Actions':<30}")
    print("-" * 60)
    for result in all_episode_results:
        print(
            f"{result['episode']:<8} {result['reward']:<8.4f} {result['steps']:<8} {result['actions']:<30}"
        )
    print("-" * 60)
    selected_actions.remove(continue_action)
    print(f"Selected {len(selected_actions)} actions: {selected_actions}")
    print("Evaluation finished.")


def save_render(env, filename, model_path):
    """Saves the rendered output if supported and requested."""
    render_dir = pathlib.Path(model_path).parent.joinpath("render")
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
        "-m", "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "-n", "--episode", type=int, default=5, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        default="data/medium",
        help="Path to dataset dir",
    )
    parser.add_argument(
        "-r", "--render", action="store_true", help="Render environment"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Create environment
    print(
        "Preloading shared DiscreteDataset to avoid redundant loading in subprocesses..."
    )
    shared_dataset = DiscreteDataset(
        data_dir=str(Path.cwd().joinpath(args.dataset_dir))
    )
    shared_dataset.load_embeddings()

    # Inject into class for SubprocVecEnv (which forks)
    DiscreteEnv.shared_dataset = shared_dataset

    # Create environment
    env = gym.make(
        config["env"]["id"],
        render_mode="human",
        random=False,
        dataset=shared_dataset,
    )

    # Evaluate
    model_path = Path.cwd().joinpath(args.model.rstrip(".zip"))
    evaluation(env, str(model_path), num_episodes=args.episode, render=args.render)


if __name__ == "__main__":
    main()
