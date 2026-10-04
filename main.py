import argparse
import torch
from run_model import Training

def parse_args():
    parser = argparse.ArgumentParser(description="Training hyperparameters for TELA")

    # Model Architecture
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=4, help="Dimension of the action space")
    parser.add_argument("--state-dim",type=int,default=19,  # 16 + 3 (states + TERM + START + PAD),
                        help="Dimension of the state space (including special tokens)")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=6,)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")

    # Training Configuration
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for optimization",)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training/eval",)
    parser.add_argument("--seq-length", type=int, default=11, help="Maximum sequence length/context window",)
    parser.add_argument("--episodes", type=int, default=32, help="Number of episodes")
    parser.add_argument("--epochs", type=int, default=32, help="Number of training epochs")

    # Special Tokens
    parser.add_argument("--start-token", type=int, default=16, help="Token index for START")
    parser.add_argument("--term-token", type=int, default=17, help="Token index for TERM")
    parser.add_argument("--pad-token", type=int, default=18, help="Token index for PAD")

    # Hardware Environment
    parser.add_argument("--device",type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on ('cuda', 'cpu', 'mps')",)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Access arguments as attributes
    print(f"Device: {args.device}")
    print(f"Learning Rate: {args.learning_rate}")  # Becomes args.learning_rate
    training_instance = Training(args)
    training_instance.run_model()