import numpy as np
import yaml


def parse_time_str(time_str: str) -> float:
    """Converts 'MM:SS' or 'HH:MM:SS' to seconds."""
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        raise ValueError("Both arrays must be non-None")
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm > 1e-6 and b_norm > 1e-6:
        return np.dot(a, b) / (a_norm * b_norm)
    return 0.0


def load_config(config_path):
    data = {}
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # complete the config

    # env
    if "env" not in data:
        data["env"] = {}
    if "id" not in data["env"]:
        data["env"]["id"] = "Discrete-v0"

    # agent
    if "agent" not in data:
        data["agent"] = {}
    if "algo" not in data["agent"]:
        data["agent"]["algo"] = "PPO"
    if "algo_kwargs" not in data["agent"]:
        data["agent"]["algo_kwargs"] = {}

    # train
    if "train" not in data:
        data["train"] = {}
    if "n_envs" not in data["train"]:
        data["train"]["n_envs"] = 8
    if "total_timesteps" not in data["train"]:
        data["train"]["total_timesteps"] = 1000000
    if "n_eval_episodes" not in data["train"]:
        data["train"]["n_eval_episodes"] = 10
    if "eval_freq" not in data["train"]:
        data["train"]["eval_freq"] = 10000
    if "seed" not in data["train"]:
        data["train"]["seed"] = 42
    if "save_path" not in data["train"]:
        data["train"]["save_path"] = "models/"
    if "log_path" not in data["train"]:
        data["train"]["log_path"] = "logs/"
    if "device" not in data["train"]:
        data["train"]["device"] = "auto"

    return data
