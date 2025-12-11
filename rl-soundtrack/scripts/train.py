import argparse
import os
from datetime import datetime
from pathlib import Path

import wandb
import yaml
from rl_soundtrack.dataset.discrete import DiscreteDataset
from rl_soundtrack.envs.discrete import DiscreteEnv
from rl_soundtrack.envs.multi_discrete import MultiDiscreteEnv
from rl_soundtrack.utils.common import load_config
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from wandb.integration.sb3.sb3 import WandbCallback

# set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Soundtrack")
    parser.add_argument("--id", type=str, default="", help="Run ID")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        default="data/medium",
        help="Path to dataset dir",
    )
    args = parser.parse_args()

    now_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_id = args.id if args.id != "" else now_time_str
    config = load_config(args.config)
    env_config = config["env"]
    agent_config = config["agent"]
    train_config = config["train"]

    model_save_path = str(Path.cwd() / train_config["save_path"] / run_id)
    train_log_path = str(Path.cwd() / train_config["log_path"] / run_id)
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(train_log_path, exist_ok=True)

    # save a copy of config with versioning
    new_config_path = Path(model_save_path).joinpath(f"{now_time_str}.yaml")
    with open(new_config_path, "w") as f:
        yaml.dump(config, f)

    # Create wandb session
    run = wandb.init(
        project="rl-soundtrack",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=run_id,
        resume="allow",
    )

    # Create environment
    print(
        "Preloading shared DiscreteDataset to avoid redundant loading in subprocesses..."
    )
    shared_dataset = DiscreteDataset(
        data_dir=str(Path.cwd().joinpath(args.dataset_dir))
    )
    shared_dataset.load_embeddings()

    # Inject into class for SubprocVecEnv (which forks)
    env_class = {"Discrete": DiscreteEnv, "MultiDiscrete": MultiDiscreteEnv}
    env_class[env_config["id"]].shared_dataset = shared_dataset

    vec_env = make_vec_env(
        env_config["id"],
        n_envs=train_config["n_envs"],
        seed=train_config["seed"],
        env_kwargs={"render_mode": "human"},
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )
    # For eval_env (single process locally usually), we can pass the dataset to save memory too
    eval_env = make_vec_env(
        env_config["id"],
        n_envs=1,
        seed=train_config["seed"],
        env_kwargs={"render_mode": "human", "dataset": shared_dataset, "random": False},
    )

    # Initialize agent
    algo_class = {"PPO": PPO, "A2C": A2C, "DQN": DQN}[agent_config["algo"]]
    if args.id != "":
        model = algo_class.load(
            f"models/{args.id}/final_model",
            env=vec_env,
            device=train_config["device"],
            tensorboard_log=train_log_path,
        )
    else:
        model = algo_class(
            env=vec_env,
            verbose=0,
            tensorboard_log=train_log_path,
            seed=train_config["seed"],
            device=train_config["device"],
            **agent_config["algo_kwargs"],
        )

    # Train agent
    wandb_callback = WandbCallback(verbose=2)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=train_config["n_eval_episodes"],
        eval_freq=max(1, train_config["eval_freq"] // train_config["n_envs"]),
        log_path=train_log_path,
        best_model_save_path=model_save_path,
        deterministic=True,
        render=False,
    )

    print(f"Starting training for {train_config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=train_config["total_timesteps"],
        reset_num_timesteps=False,
        callback=[wandb_callback, eval_callback],
    )

    # Save final model
    final_model_path = Path(model_save_path).joinpath("final_model")
    model.save(final_model_path)
    print("Training finished and model saved.")

    run.finish()


if __name__ == "__main__":
    main()
