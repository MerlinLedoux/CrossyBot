"""
trainer.py — Boucle d'entraînement PPO pour CrossyBot.

Flux par itération :
  1. Collecte  : joue n_steps steps sur n_envs envs parallèles
  2. GAE       : calcule les avantages dans le buffer
  3. Update    : n_epochs passes PPO sur le buffer
  4. Reset     : vide le buffer, log les métriques
"""
import time
import torch
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from .network import ActorCritic, OBS_SIZE
from .rollout import RolloutBuffer
from .ppo     import PPO


def _make_env(seed_offset: int = 0):
    """Factory pour un env Crossy — chaque env a sa propre seed."""
    def _init():
        from training.env.crossy_env import CrossyEnv
        env = CrossyEnv()
        env.reset(seed=int(time.time() * 1000) % 100000 + seed_offset)
        return env
    return _init


class Trainer:

    def __init__(
        self,
        n_envs:         int   = 16,
        n_steps:        int   = 512,
        n_epochs:       int   = 4,
        batch_size:     int   = 256,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        clip_range:     float = 0.2,
        vf_coef:        float = 0.5,
        ent_coef:       float = 0.01,
        max_grad_norm:  float = 0.5,
        save_every:     int   = 50,    # sauvegarde tous les N updates
        save_path:      str   = "training/models/crossybot.pt",
        log_every:      int   = 10,    # affiche les métriques tous les N updates
    ):
        self.n_envs    = n_envs
        self.n_steps   = n_steps
        self.save_every = save_every
        self.save_path  = save_path
        self.log_every  = log_every

        # --- environnements parallèles ---
        self.envs = AsyncVectorEnv([_make_env(i) for i in range(n_envs)])

        # --- réseau et algorithme ---
        self.network = ActorCritic()
        self.ppo     = PPO(
            network       = self.network,
            lr            = lr,
            n_epochs      = n_epochs,
            batch_size    = batch_size,
            clip_range    = clip_range,
            vf_coef       = vf_coef,
            ent_coef      = ent_coef,
            max_grad_norm = max_grad_norm,
        )

        # --- buffer ---
        self.buffer = RolloutBuffer(
            n_steps    = n_steps,
            n_envs     = n_envs,
            obs_dim    = OBS_SIZE,
            gamma      = gamma,
            gae_lambda = gae_lambda,
        )

        # --- métriques de suivi ---
        self.update_count   = 0
        self.total_steps    = 0
        self.episode_rewards: list[float] = []   # récompenses des épisodes terminés

    # --- boucle principale ---------------------------------------------------

    def train(self, total_updates: int = 1000) -> None:
        """Lance l'entraînement pour `total_updates` mises à jour PPO."""
        obs_np, _ = self.envs.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32)   # (n_envs, obs_dim)

        # Récompenses cumulées par env (pour le logging)
        ep_rewards = np.zeros(self.n_envs)

        print(f"Démarrage entraînement — {total_updates} updates × "
              f"{self.n_steps} steps × {self.n_envs} envs = "
              f"{total_updates * self.n_steps * self.n_envs:,} steps total\n")

        for update in range(1, total_updates + 1):
            t_start = time.time()

            # ----------------------------------------------------------------
            # 1. COLLECTE
            # ----------------------------------------------------------------
            self.buffer.reset()

            for _ in range(self.n_steps):
                action, log_prob, value = self.network.act(obs)

                obs_np, reward_np, terminated_np, truncated_np, _ = \
                    self.envs.step(action.numpy())

                done_np = terminated_np | truncated_np
                reward  = torch.tensor(reward_np,   dtype=torch.float32)
                done    = torch.tensor(done_np,      dtype=torch.float32)

                self.buffer.add(obs, action, log_prob, reward, done, value)

                # Logging des épisodes terminés
                ep_rewards += reward_np
                for i, d in enumerate(done_np):
                    if d:
                        self.episode_rewards.append(ep_rewards[i])
                        ep_rewards[i] = 0.0

                obs = torch.tensor(obs_np, dtype=torch.float32)
                self.total_steps += self.n_envs

            # ----------------------------------------------------------------
            # 2. GAE
            # ----------------------------------------------------------------
            with torch.no_grad():
                _, _, last_values = self.network.act(obs)
            self.buffer.compute_returns(last_values)

            # ----------------------------------------------------------------
            # 3. MISE À JOUR PPO
            # ----------------------------------------------------------------
            metrics = self.ppo.update(self.buffer)
            self.update_count += 1

            # ----------------------------------------------------------------
            # 4. LOGGING ET SAUVEGARDE
            # ----------------------------------------------------------------
            if update % self.log_every == 0:
                self._log(update, total_updates, metrics, t_start)

            if update % self.save_every == 0:
                self.save(self.save_path)

        # Sauvegarde finale
        self.save(self.save_path)
        print(f"\nEntraînement terminé. Modèle sauvegardé → {self.save_path}")

    # --- utilitaires ---------------------------------------------------------

    def _log(self, update: int, total: int, metrics: dict, t_start: float) -> None:
        elapsed  = time.time() - t_start
        fps      = self.n_steps * self.n_envs / elapsed

        recent   = self.episode_rewards[-100:] if self.episode_rewards else [0]
        mean_rew = sum(recent) / len(recent)
        max_rew  = max(recent)

        print(
            f"Update {update:4d}/{total} | "
            f"steps {self.total_steps:>9,} | "
            f"fps {fps:>6.0f} | "
            f"rew_mean {mean_rew:>7.2f} | "
            f"rew_max {max_rew:>7.2f} | "
            f"L_pol {metrics['loss_policy']:>7.4f} | "
            f"L_val {metrics['loss_value']:>7.4f} | "
            f"entropy {-metrics['loss_entropy']:>6.4f} | "
            f"clip {metrics['clip_frac']:>5.3f}"
        )

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state":    self.network.state_dict(),
            "optimizer_state":  self.ppo.optimizer.state_dict(),
            "update_count":     self.update_count,
            "total_steps":      self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.network.load_state_dict(checkpoint["network_state"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint["update_count"]
        self.total_steps  = checkpoint["total_steps"]
        print(f"Checkpoint chargé — update {self.update_count}, "
              f"steps {self.total_steps:,}")
