from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    """

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        """
        Get the action for the given observation.
        """
        pass

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval=None,
        tb_log_name="run",
        reset_num_timesteps=True,
        progress_bar=False,
    ):
        """
        Train the agent.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the agent's model.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the agent's model.
        """
        pass
