from gymnasium.vector import AsyncVectorEnv
from env.crossy_env import CrossyEnv

def make_env():
    def _init():
        return CrossyEnv()
    return _init

class Trainer:
    def __init__(self, n_envs=16):
        self.envs = AsyncVectorEnv([make_env() for _ in range(n_envs)])
        self.agent = PPO(...)

    def train(self):
        obs, _ = self.envs.reset()
        # boucle d'entraînement...