"""
Loss totale :
    L = L_policy + c1 * L_value - c2 * L_entropy

    L_policy  = -E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]
    L_value   = 0.5 * MSE( V(s), R_t )
    L_entropy = -H[pi]   (maximiser l'entropie → exploration)

    r_t = pi(a|s) / pi_old(a|s)  — ratio de probabilités
"""
import torch
import torch.nn as nn
from .network import ActorCritic
from .rollout import RolloutBuffer


class PPO:

    def __init__(
        self,
        network:        ActorCritic,    # Architecture utiliser pour prendre une décision en fonction des observation.
        lr:             float = 3e-4,   # Learning rate agit sur les poids du réseaux pour qu'on ne face pas de changement trop brusque.
        n_epochs:       int   = 4,      # Nombre de fois ou l'on passe sur la même donner collecter pour mettre ajour le réseau.
        batch_size:     int   = 256,    # Taille d'un batch pour la mise a jour des poids.
        clip_range:     float = 0.2,    # Définie la varariation maximal pour la polique permet de lisser l'apprentissage. Même role que le lr mais a un autre niveau.
        vf_coef:        float = 0.5,    # Value fonction coefficient est le poids de la loss du critic dans la loss totale
        ent_coef:       float = 0.01,   # Définie a qu'elle point on encourage l'exploration
        max_grad_norm:  float = 0.5,    # contrôle la direction du pas Il ne touche pas au learning rate. Il regarde la norme globale du vecteur gradient, et si elle est trop grande, il redimensionne le vecteur entier pour que sa longueur fasse exactement 0.5 — sans changer sa direction.

    ):
        self.network       = network
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.clip_range    = clip_range
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Effectue n_epochs passes sur le buffer.
        Retourne un dictionnaire de métriques pour le logging.
        """
        metrics = {
            "loss_policy":  [], # 
            "loss_value":   [], # 
            "loss_entropy": [], # 
            "loss_total":   [], # 
            "clip_frac":    [], # fraction de ratios clippés
        }

        for _ in range(self.n_epochs):
            for obs, actions, old_log_probs, advantages, returns in \
                    buffer.get_batches(self.batch_size):

                # --- recalcul des probabilités et valeurs actuelles ---
                log_probs, entropy, values = self.network.evaluate(obs, actions)

                # --- ratio pi / pi_old ---
                ratio = torch.exp(log_probs - old_log_probs)

                # --- policy loss (clipped surrogate) ---
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                           1.0 + self.clip_range) * advantages
                loss_policy = -torch.min(surr1, surr2).mean()

                # --- value loss ---
                loss_value = 0.5 * nn.functional.mse_loss(values, returns)

                # --- entropy loss (négatif car on maximise) ---
                loss_entropy = -entropy.mean()

                # --- loss totale ---
                loss = loss_policy + self.vf_coef * loss_value + self.ent_coef * loss_entropy

                # --- mise à jour des poids ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- métriques ---
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean()
                metrics["loss_policy"].append(loss_policy.item())
                metrics["loss_value"].append(loss_value.item())
                metrics["loss_entropy"].append(loss_entropy.item())
                metrics["loss_total"].append(loss.item())
                metrics["clip_frac"].append(clip_frac.item())

        # Moyenne sur tous les mini-batchs et toutes les epochs
        return {k: sum(v) / len(v) for k, v in metrics.items()}
