"""
train.py — Point d'entrée de l'entraînement CrossyBot.

Usage :
    python train.py
    python train.py --updates 2000 --envs 32 --steps 1024
"""
import argparse
from training.agent.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement PPO CrossyBot")
    p.add_argument("--updates",    type=int,   default=1000,  help="Nombre de mises à jour PPO")
    p.add_argument("--envs",       type=int,   default=16,    help="Nombre d'environnements parallèles")
    p.add_argument("--steps",      type=int,   default=512,   help="Steps collectés par env par update")
    p.add_argument("--lr",         type=float, default=3e-4,  help="Learning rate")
    p.add_argument("--epochs",     type=int,   default=4,     help="Epochs PPO par update")
    p.add_argument("--batch",      type=int,   default=256,   help="Taille des mini-batchs")
    p.add_argument("--load",       type=str,   default=None,  help="Chemin d'un checkpoint à charger")
    p.add_argument("--save",       type=str,   default="training/models/crossybot.pt")
    p.add_argument("--no-wandb",   action="store_true",      help="Désactive WandB")
    p.add_argument("--run-name",   type=str,   default=None,  help="Nom de la run WandB")
    return p.parse_args()


def main():
    args = parse_args()

    trainer = Trainer(
        n_envs         = args.envs,
        n_steps        = args.steps,
        n_epochs       = args.epochs,
        batch_size     = args.batch,
        lr             = args.lr,
        save_path      = args.save,
        use_wandb      = not args.no_wandb,
        wandb_run_name = args.run_name,
    )

    if args.load:
        trainer.load(args.load)

    trainer.train(total_updates=args.updates)


if __name__ == "__main__":
    main()
