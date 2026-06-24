"""
export_model.py — Exporte les poids du modèle entraîné en JSON
pour l'inférence directe dans la version web (sans dépendance externe).

Les poids sont aplatis en row-major (identique au layout PyTorch)
et chargés par web/src/ai/agent.js qui reconstruit le réseau en JS pur.

Usage :
    python export_model.py
    python export_model.py --model training/models/crossybot.pt
    python export_model.py --model training/models/crossybot.pt --output web/assets/crossybot.json
"""
import argparse
import json
import torch
from training.agent.network import ActorCritic, OBS_SIZE, N_ACTIONS


def export(model_path: str, output_path: str) -> None:
    network = ActorCritic()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    network.load_state_dict(ckpt["network_state"])
    network.eval()

    def layer(l):
        """Sérialise une nn.Linear en dict JSON."""
        return {
            "weight":       l.weight.detach().numpy().flatten().tolist(),
            "bias":         l.bias.detach().numpy().tolist(),
            "in_features":  l.in_features,
            "out_features": l.out_features,
        }

    payload = {
        "obs_size":         OBS_SIZE,
        "n_actions":        N_ACTIONS,
        "lane_encoder_fc1": layer(network.lane_encoder.net[0]),  # Linear(11, 64)
        "lane_encoder_fc2": layer(network.lane_encoder.net[2]),  # Linear(64, 32)
        "trunk_fc1":        layer(network.trunk[0]),              # Linear(162, 256)
        "trunk_fc2":        layer(network.trunk[2]),              # Linear(256, 128)
        "policy_head":      layer(network.policy_head),           # Linear(128, 5)
    }

    with open(output_path, "w") as f:
        json.dump(payload, f)

    total = sum(
        len(v["weight"]) + len(v["bias"])
        for v in payload.values() if isinstance(v, dict)
    )
    print(f"✓ Modèle exporté : {output_path}  ({total:,} paramètres)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CrossyBot model weights to JSON")
    parser.add_argument("--model",  default="training/models/crossybot.pt",
                        help="Chemin vers le checkpoint PyTorch (.pt)")
    parser.add_argument("--output", default="web/assets/crossybot.json",
                        help="Chemin de sortie JSON")
    args = parser.parse_args()
    export(args.model, args.output)
