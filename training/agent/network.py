"""
network.py — Architecture ActorCritic pour CrossyBot.

Structure :
  - LaneEncoder  : MLP partagé appliqué indépendamment à chaque lane
  - Tronc commun : MLP partagé Actor/Critic
  - PolicyHead   : logits → distribution Categorical
  - ValueHead    : scalaire V(s)
"""
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Dimensions issues de l'environnement (doit correspondre à crossy_env.py)
N_LANES      = 5    # OBS_LANES  : 1 derrière + joueur + 3 devant
LANE_FEAT    = 11   # type + speed + 9 colonnes jouables (OBS_PLAYABLE_W)
PLAYER_FEAT  = 2    # player_x normalisé, position verticale normalisée
OBS_SIZE     = N_LANES * LANE_FEAT + PLAYER_FEAT   # 5*11+2 = 57
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



class ActorCritic(nn.Module):

    LANE_EMBED   = 32
    TRUNK_HIDDEN = 256
    TRUNK_OUT    = 128

    def __init__(self):
        super().__init__()

        self.lane_encoder = LaneEncoder(LANE_FEAT, self.LANE_EMBED)

        trunk_in = self.LANE_EMBED * N_LANES + PLAYER_FEAT   # 32*5 + 2 = 162

        self.trunk = nn.Sequential(
            _orthogonal(nn.Linear(trunk_in, self.TRUNK_HIDDEN)),
            nn.ReLU(),
            _orthogonal(nn.Linear(self.TRUNK_HIDDEN, self.TRUNK_OUT)),
            nn.ReLU(),
        )

        self.policy_head = _orthogonal(nn.Linear(self.TRUNK_OUT, N_ACTIONS), gain=0.01)
        self.value_head  = _orthogonal(nn.Linear(self.TRUNK_OUT, 1),         gain=1.0)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        lanes_flat = obs[:, :N_LANES * LANE_FEAT]          # (B, 55)
        player_pos = obs[:, N_LANES * LANE_FEAT:]          # (B, 2)

        lanes      = lanes_flat.reshape(-1, N_LANES, LANE_FEAT)   # (B, 5, 11)
        lane_embed = self.lane_encoder(lanes)                      # (B, 5, 32)
        flat       = lane_embed.reshape(lane_embed.size(0), -1)    # (B, 160)

        x = torch.cat([flat, player_pos], dim=1)                   # (B, 162)
        return self.trunk(x)                                        # (B, 128)

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
