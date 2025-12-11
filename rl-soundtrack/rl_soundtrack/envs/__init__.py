from gymnasium.envs.registration import register
from rl_soundtrack.envs.discrete import DiscreteEnv
from rl_soundtrack.envs.multi_discrete import MultiDiscreteEnv

register(
    id="Discrete",
    entry_point="rl_soundtrack.envs:DiscreteEnv",
)
register(
    id="MultiDiscrete",
    entry_point="rl_soundtrack.envs:MultiDiscreteEnv",
)
