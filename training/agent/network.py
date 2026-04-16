"""
network.py — Architecture ActorCritic pour CrossyBot.

Structure :
  - LaneEncoder  : MLP partagé appliqué indépendamment à chaque lane
  - SpatialConv  : Conv1D pour capturer les relations entre lanes consécutives
  - Tronc commun : MLP partagé Actor/Critic
  - PolicyHead   : logits → distribution Categorical
  - ValueHead    : scalaire V(s)
"""
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Dimensions issues de l'environnement
N_LANES      = 13   # GRID_H
LANE_FEAT    = 15   # type + speed + 13 valeurs d'obstacle (GRID_W)
PLAYER_FEAT  = 2    # player_x normalisé, position verticale normalisée
OBS_SIZE     = N_LANES * LANE_FEAT + PLAYER_FEAT   # 197
N_ACTIONS    = 5


def _orthogonal(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    """Initialisation orthogonale — détail PPO critique pour la stabilité."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class LaneEncoder(nn.Module):
    """
    MLP partagé appliqué indépendamment à chaque lane.
    Entrée  : (batch, N_LANES, LANE_FEAT)
    Sortie  : (batch, N_LANES, lane_embed_dim)

    Le partage des poids encode l'hypothèse que toutes les lanes
    obéissent aux mêmes règles physiques, peu importe leur position.
    """

    def __init__(self, in_dim: int = LANE_FEAT, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal(nn.Linear(in_dim, 64)),
            nn.ReLU(),
            _orthogonal(nn.Linear(64, embed_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, N_LANES, LANE_FEAT)
        # on applique le MLP à chaque lane indépendamment
        batch, n_lanes, feat = x.shape
        out = self.net(x.reshape(batch * n_lanes, feat))     # (batch*N, embed)
        return out.reshape(batch, n_lanes, -1)               # (batch, N, embed)


class SpatialConv(nn.Module):
    """
    Conv1D sur la dimension des lanes pour capturer les groupes consécutifs
    (ex : rivière multi-lignes, groupe de routes).
    Entrée  : (batch, embed_dim, N_LANES)
    Sortie  : (batch, conv_dim * N_LANES)  — flatten conserve la position
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out_dim = out_channels * N_LANES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, in_channels, N_LANES)
        out = self.conv(x)          # (batch, out_channels, N_LANES)
        return out.flatten(1)       # (batch, out_channels * N_LANES)


class ActorCritic(nn.Module):
    """
    Réseau Actor-Critic complet pour PPO.

    Flux :
      obs (197,)
        ├─ lanes (13, 15) → LaneEncoder → SpatialConv → trunk_in (832,)
        └─ player_pos (2,)
              └─ concat → (834,) → tronc partagé (256 → 128)
                    ├─ policy_head → logits (5,)
                    └─ value_head  → V(s)  (1,)
    """

    LANE_EMBED  = 32
    CONV_CH     = 64
    TRUNK_HIDDEN = 256
    TRUNK_OUT    = 128

    def __init__(self):
        super().__init__()

        self.lane_encoder = LaneEncoder(LANE_FEAT, self.LANE_EMBED)
        self.spatial_conv  = SpatialConv(self.LANE_EMBED, self.CONV_CH)

        conv_flat = self.CONV_CH * N_LANES          # 64 * 13 = 832
        trunk_in  = conv_flat + PLAYER_FEAT         # 832 + 2 = 834

        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(trunk_in, self.TRUNK_HIDDEN)),
            nn.ReLU(),
            _orthogonal(nn.Linear(self.TRUNK_HIDDEN, self.TRUNK_OUT)),
            nn.ReLU(),
        )

        # Têtes avec gains réduits (logits proches de 0 au départ = exploration uniforme)
        self.policy_head = _orthogonal(nn.Linear(self.TRUNK_OUT, N_ACTIONS), gain=0.01)
        self.value_head  = _orthogonal(nn.Linear(self.TRUNK_OUT, 1),         gain=1.0)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Extrait les features partagées depuis l'observation brute."""
        lanes_flat  = obs[:, :N_LANES * LANE_FEAT]                        # (B, 195)
        player_pos  = obs[:, N_LANES * LANE_FEAT:]                        # (B, 2)

        lanes       = lanes_flat.view(-1, N_LANES, LANE_FEAT)             # (B, 13, 15)
        lane_embed  = self.lane_encoder(lanes)                             # (B, 13, 32)
        spatial     = self.spatial_conv(lane_embed.permute(0, 2, 1))      # (B, 832)

        x = torch.cat([spatial, player_pos], dim=1)                       # (B, 834)
        return self.trunk(x)                                               # (B, 128)

    def forward(self, obs: torch.Tensor):
        """Retourne (distribution, valeur) — utilisé à l'entraînement."""
        features = self._encode(obs)
        dist  = Categorical(logits=self.policy_head(features))
        value = self.value_head(features).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """
        Échantillonne une action et retourne (action, log_prob, valeur).
        Utilisé lors de la collecte des rollouts (pas de gradient).
        """
        features  = self._encode(obs)
        dist      = Categorical(logits=self.policy_head(features))
        action    = dist.sample()
        log_prob  = dist.log_prob(action)
        value     = self.value_head(features).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Recalcule log_prob, entropie et valeur sur un batch stocké.
        Utilisé lors de la mise à jour PPO (avec gradient).
        """
        features  = self._encode(obs)
        dist      = Categorical(logits=self.policy_head(features))
        log_prob  = dist.log_prob(actions)
        entropy   = dist.entropy()
        value     = self.value_head(features).squeeze(-1)
        return log_prob, entropy, value
