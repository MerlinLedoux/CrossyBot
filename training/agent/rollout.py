"""
rollout.py — Buffer de collecte des expériences pour PPO.

Stocke N steps * M envs d'expériences, puis calcule les avantages
via GAE (Generalized Advantage Estimation, Schulman 2015).

GAE :
    delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t       = sum_{k=0}^{inf} (gamma * lambda)^k * delta_t+k

    - gamma  : discount (horizon temporel)
    - lambda : biais/variance trade-off (0 = TD pur, 1 = Monte Carlo pur)
"""
import torch
import numpy as np


class RolloutBuffer:
    """
    Stocke les transitions collectées sur n_steps * n_envs steps.
    Toutes les données sont des tenseurs CPU.
    """

    def __init__(
        self,
        n_steps:  int,
        n_envs:   int,
        obs_dim:  int,
        gamma:    float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps     = n_steps
        self.n_envs      = n_envs
        self.obs_dim     = obs_dim
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda

        self._init_storage()

    # initialisation 
    def _init_storage(self) -> None:
        T, E, D = self.n_steps, self.n_envs, self.obs_dim
        self.obs       = torch.zeros(T, E, D)
        self.actions   = torch.zeros(T, E, dtype=torch.long)
        self.log_probs = torch.zeros(T, E)
        self.rewards   = torch.zeros(T, E)
        self.dones     = torch.zeros(T, E)
        self.values    = torch.zeros(T, E)
        self.ptr       = 0          # pointeur sur le step courant
        self.full      = False

    def reset(self) -> None:
        self._init_storage()

    # insertion 
    def add(
        self,
        obs:      torch.Tensor,   # (n_envs, obs_dim)
        action:   torch.Tensor,   # (n_envs,)
        log_prob: torch.Tensor,   # (n_envs,)
        reward:   torch.Tensor,   # (n_envs,)
        done:     torch.Tensor,   # (n_envs,)  float 0/1
        value:    torch.Tensor,   # (n_envs,)
    ) -> None:
        assert self.ptr < self.n_steps, "Buffer plein — appeler compute_returns() puis reset()"
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.values[self.ptr]    = value
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    # calcul GAE 
    def compute_returns(self, last_values: torch.Tensor) -> None:
        """
        Calcule les avantages GAE et les returns cibles pour la value function.
        Doit être appelé une fois le buffer rempli, avant de lire les données.

        last_values : (n_envs,) — V(s_{T}) estimée sur la dernière obs
        """
        assert self.full, "Buffer pas encore plein"

        advantages = torch.zeros_like(self.rewards)
        last_gae   = torch.zeros(self.n_envs)

        # Parcours en arrière pour accumuler GAE
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values       = self.values[t + 1]

            delta    = (self.rewards[t]
                        + self.gamma * next_values * next_non_terminal
                        - self.values[t])
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + self.values   # cibles pour la value head

    # itération mini-batchs 
    def get_batches(self, batch_size: int):
        """
        Génère des mini-batchs aléatoires à partir des données collectées.
        Normalise les avantages par mini-batch (détail PPO important).

        Yields : (obs, actions, old_log_probs, advantages, returns)
        """
        assert hasattr(self, "advantages"), "Appeler compute_returns() d'abord"

        total = self.n_steps * self.n_envs

        # Aplatit les dimensions (n_steps, n_envs) en (total,)
        obs_flat       = self.obs.view(total, self.obs_dim)
        actions_flat   = self.actions.view(total)
        log_probs_flat = self.log_probs.view(total)
        advantages_flat= self.advantages.view(total)
        returns_flat   = self.returns.view(total)

        # Mélange aléatoire
        indices = torch.randperm(total)

        for start in range(0, total, batch_size):
            idx = indices[start: start + batch_size]

            adv_batch = advantages_flat[idx]
            # Normalisation par mini-batch : réduit la variance, stabilise l'entraînement
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

            yield (
                obs_flat[idx],
                actions_flat[idx],
                log_probs_flat[idx],
                adv_batch,
                returns_flat[idx],
            )
