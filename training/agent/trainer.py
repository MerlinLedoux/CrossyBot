"""
trainer.py — Boucle d'entraînement PPO pour CrossyBot avec suivi WandB.

Flux par itération :
  1. Collecte  : joue n_steps steps sur n_envs envs parallèles
  2. GAE       : calcule les avantages dans le buffer
  3. Update    : n_epochs passes PPO sur le buffer
  4. Log       : envoie les métriques à WandB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUIDE WANDB — QU'EST-CE QUE C'EST ?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WandB (Weights & Biases) est un outil de suivi d'expériences ML.
Il enregistre automatiquement toutes tes métriques (loss, reward, etc.)
et les affiche sur un dashboard web en temps réel.

INSTALLATION :
    pip install wandb
    wandb login          ← ouvre le navigateur, tu crées un compte gratuit
                           et tu colles ton API key

CONCEPTS CLÉS :
  - Project  : ensemble d'expériences liées (ex: "crossybot")
  - Run      : une session d'entraînement (un appel à train.py)
  - Entity   : ton nom d'utilisateur WandB
  - Step     : l'axe X de tous les graphiques (ici = total_steps)

TABLEAU DE BORD :
    https://wandb.ai/<ton_username>/<nom_du_projet>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import time
import torch
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from .network import ActorCritic, OBS_SIZE
from .rollout import RolloutBuffer
from .ppo     import PPO

# ─────────────────────────────────────────────────────────────────────
# Import WandB — on l'importe ici pour pouvoir désactiver facilement
# en passant use_wandb=False au Trainer sans modifier le reste du code.
# ─────────────────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
        save_every:     int   = 50,
        save_path:      str   = "training/models/crossybot.pt",
        log_every:      int   = 10,
        # ── paramètres WandB ──────────────────────────────────────────
        use_wandb:      bool  = True,
        wandb_project:  str   = "crossybot",
        # wandb_entity  : ton username WandB (None = détecté automatiquement)
        wandb_entity:   str   = None,
        # wandb_run_name : nom affiché dans le dashboard pour cette run.
        # None = WandB génère un nom aléatoire poétique (ex: "dashing-moon-42")
        wandb_run_name: str   = None,
    ):
        self.n_envs     = n_envs
        self.n_steps    = n_steps
        self.save_every = save_every
        self.save_path  = save_path
        self.log_every  = log_every
        self.use_wandb  = use_wandb and WANDB_AVAILABLE

        # Stocke les hyperparamètres pour les passer à WandB
        self.hparams = dict(
            n_envs=n_envs, n_steps=n_steps, n_epochs=n_epochs,
            batch_size=batch_size, lr=lr, gamma=gamma,
            gae_lambda=gae_lambda, clip_range=clip_range,
            vf_coef=vf_coef, ent_coef=ent_coef, max_grad_norm=max_grad_norm,
        )

        # ── initialisation WandB ──────────────────────────────────────
        # wandb.init() démarre une nouvelle "run".
        # Tout ce qui est loggé après cet appel apparaît dans le dashboard.
        #
        # Paramètres importants :
        #   project  : nom du projet (crée automatiquement s'il n'existe pas)
        #   entity   : ton username (optionnel)
        #   name     : nom de cette run dans le dashboard
        #   config   : dictionnaire des hyperparamètres → affiché dans
        #              l'onglet "Config" et utilisable pour filtrer/comparer
        #              des runs entre elles
        #   resume   : "allow" permet de reprendre une run interrompue
        #              si le même run_id est retrouvé
        # ─────────────────────────────────────────────────────────────
        if self.use_wandb:
            wandb.init(
                project = wandb_project,
                entity  = wandb_entity,
                name    = wandb_run_name,
                config  = self.hparams,   # ← hyperparamètres visibles sur le dashboard
                resume  = "allow",
            )

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

        # ── wandb.watch() ─────────────────────────────────────────────
        # Surveille automatiquement les gradients et les poids du réseau.
        # Cela ajoute dans le dashboard :
        #   - des histogrammes des gradients par couche
        #   - des histogrammes des poids par couche
        # Utile pour détecter les problèmes de vanishing/exploding gradients.
        #
        # log_freq : fréquence de log en nombre de backward passes
        # log      : "gradients" | "parameters" | "all"
        # ─────────────────────────────────────────────────────────────
        if self.use_wandb:
            wandb.watch(self.network, log="all", log_freq=100)

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
        self.episode_rewards: list[float] = []
        self.episode_scores:  list[float] = []   # score max atteint par épisode

    # --- boucle principale ---------------------------------------------------

    def train(self, total_updates: int = 1000) -> None:
        obs_np, _ = self.envs.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32)

        ep_rewards = np.zeros(self.n_envs)
        ep_scores  = np.zeros(self.n_envs)

        print(f"Démarrage entraînement — {total_updates} updates × "
              f"{self.n_steps} steps × {self.n_envs} envs = "
              f"{total_updates * self.n_steps * self.n_envs:,} steps total\n")

        if self.use_wandb:
            # Affiche dans le terminal l'URL du dashboard de cette run.
            # Tu peux cliquer dessus pour ouvrir directement la page WandB.
            print(f"Dashboard WandB : {wandb.run.url}\n")

        for update in range(1, total_updates + 1):
            t_start = time.time()

            # ----------------------------------------------------------------
            # 1. COLLECTE
            # ----------------------------------------------------------------
            self.buffer.reset()

            for _ in range(self.n_steps):
                action, log_prob, value = self.network.act(obs)

                obs_np, reward_np, terminated_np, truncated_np, infos = \
                    self.envs.step(action.numpy())

                done_np = terminated_np | truncated_np
                reward  = torch.tensor(reward_np, dtype=torch.float32)
                done    = torch.tensor(done_np,   dtype=torch.float32)

                self.buffer.add(obs, action, log_prob, reward, done, value)

                ep_rewards += reward_np
                for i, d in enumerate(done_np):
                    if d:
                        self.episode_rewards.append(float(ep_rewards[i]))
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
            # 4. LOGGING
            # ----------------------------------------------------------------
            if update % self.log_every == 0:
                self._log(update, total_updates, metrics, t_start)

            if update % self.save_every == 0:
                self.save(self.save_path)

        self.save(self.save_path)
        print(f"\nEntraînement terminé. Modèle sauvegardé → {self.save_path}")

        # ── wandb.finish() ────────────────────────────────────────────
        # Termine proprement la run WandB.
        # IMPORTANT : sans cet appel, la run reste en état "running"
        # sur le dashboard même après la fin du script.
        # ─────────────────────────────────────────────────────────────
        if self.use_wandb:
            wandb.finish()

    # --- logging -------------------------------------------------------------

    def _log(self, update: int, total: int, metrics: dict, t_start: float) -> None:
        elapsed  = time.time() - t_start
        fps      = self.n_steps * self.n_envs / elapsed

        recent   = self.episode_rewards[-100:] if self.episode_rewards else [0.0]
        mean_rew = float(np.mean(recent))
        std_rew  = float(np.std(recent))
        max_rew  = float(np.max(recent))
        min_rew  = float(np.min(recent))

        # ── affichage terminal ────────────────────────────────────────
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

        # ── wandb.log() ───────────────────────────────────────────────
        # C'est la fonction centrale de WandB.
        # Elle prend un dictionnaire {nom_metrique: valeur} et l'envoie
        # au serveur WandB.
        #
        # Chaque clé devient un graphique séparé dans le dashboard.
        # Tu peux organiser les graphiques en groupes avec le préfixe
        # "groupe/nom" (ex: "perfs/reward_mean" apparaît dans le groupe "perfs").
        #
        # step : l'axe X des graphiques. On utilise total_steps pour
        #        que les courbes soient comparables entre des runs avec
        #        des fréquences de log différentes.
        # ─────────────────────────────────────────────────────────────
        if self.use_wandb:
            wandb.log({

                # ── Récompenses ───────────────────────────────────────
                # Ces courbes montrent si l'agent progresse.
                # reward_mean doit augmenter au fil du temps.
                "perfs/reward_mean":   mean_rew,
                "perfs/reward_max":    max_rew,
                "perfs/reward_min":    min_rew,
                "perfs/reward_std":    std_rew,
                "perfs/episodes_done": len(self.episode_rewards),

                # ── Losses PPO ────────────────────────────────────────
                # loss_policy  : doit rester stable et faible
                # loss_value   : doit diminuer (le critic apprend V(s))
                # entropy      : doit rester > 0 (exploration maintenue)
                #                Si elle tombe à 0 : l'agent est coincé
                #                dans une stratégie déterministe sous-optimale
                "losses/policy":   metrics["loss_policy"],
                "losses/value":    metrics["loss_value"],
                "losses/entropy":  -metrics["loss_entropy"],   # positif = lisible
                "losses/total":    metrics["loss_total"],

                # ── Santé du clipping PPO ─────────────────────────────
                # clip_frac : fraction des ratios qui ont été clippés.
                # Valeur saine : entre 0.05 et 0.30
                #   < 0.05 : le lr est trop petit, les updates sont minuscules
                #   > 0.30 : le lr est trop grand, les updates sont trop agressives
                "ppo/clip_fraction": metrics["clip_frac"],

                # ── Vitesse d'entraînement ────────────────────────────
                # fps : frames (steps) par seconde
                # Utile pour comparer l'efficacité de différentes configs
                "perf/fps":          fps,
                "perf/update":       update,

            }, step=self.total_steps)
            # ↑ step=total_steps : l'axe X de tous les graphiques est
            #   le nombre de steps d'environnement, pas le numéro d'update.
            #   Cela rend les courbes comparables entre runs avec
            #   des valeurs de n_steps différentes.

    # --- sauvegarde ----------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network_state":    self.network.state_dict(),
            "optimizer_state":  self.ppo.optimizer.state_dict(),
            "update_count":     self.update_count,
            "total_steps":      self.total_steps,
        }, path)

        # ── wandb.save() ──────────────────────────────────────────────
        # Uploade le fichier de checkpoint vers WandB.
        # Il sera stocké dans l'onglet "Files" de la run sur le dashboard
        # et téléchargeable depuis n'importe où.
        #
        # Utile pour :
        #   - partager un modèle entraîné
        #   - reprendre l'entraînement sur une autre machine
        #   - archiver les checkpoints liés à une run précise
        # ─────────────────────────────────────────────────────────────
        if self.use_wandb:
            wandb.save(path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.network.load_state_dict(checkpoint["network_state"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update_count = checkpoint["update_count"]
        self.total_steps  = checkpoint["total_steps"]
        print(f"Checkpoint chargé — update {self.update_count}, "
              f"steps {self.total_steps:,}")
