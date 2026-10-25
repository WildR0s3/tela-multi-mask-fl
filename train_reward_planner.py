import argparse
import os

import gymnasium as gym
import torch

from logger.logging_module import initilaize_logger, log
from maps import Maps
from reward_planner import RewardPlanner


def parse_args():
    parser = argparse.ArgumentParser(description="Train the independent reward planner")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=19)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--start-token", type=int, default=16)
    parser.add_argument("--term-token", type=int, default=17)
    parser.add_argument("--pad-token", type=int, default=18)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu", "mps"),
    )
    return parser.parse_args()


class RewardPlannerTraining:
    EDGE_STEPS = 4
    EPISODE_STEPS = 16

    def __init__(self, args):
        self.args = args
        self.env = gym.make(
            "FrozenLake-v1",
            desc=Maps.clean_4x4,
            max_episode_steps=self.EPISODE_STEPS,
            is_slippery=False,
        )

    def collect_sequence(self):
        args = self.args
        states = torch.full(
            (RewardPlanner.SEQUENCE_LENGTH,), args.pad_token, dtype=torch.long
        )
        rewards = torch.full_like(states, args.pad_token)
        states[0] = args.start_token
        rewards[0] = args.start_token

        state, _ = self.env.reset()
        states[1] = state
        rewards[1] = RewardPlanner.NO_REWARD_TOKEN
        planner_position = 2
        edge_steps = 0
        edge_rewarded = False

        for _ in range(self.EPISODE_STEPS):
            action = int(torch.randint(args.action_dim, ()).item())
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            edge_steps += 1
            edge_rewarded = edge_rewarded or reward > 0
            done = terminated or truncated

            if edge_steps == self.EDGE_STEPS or done:
                states[planner_position] = next_state
                rewards[planner_position] = (
                    RewardPlanner.REWARD_TOKEN if edge_rewarded else RewardPlanner.NO_REWARD_TOKEN
                )
                planner_position += 1
                edge_steps = 0
                edge_rewarded = False

            if done:
                break

        if planner_position < RewardPlanner.SEQUENCE_LENGTH:
            states[planner_position] = args.term_token
            rewards[planner_position] = args.term_token
            planner_position += 1
        states[planner_position:] = args.pad_token
        rewards[planner_position:] = args.pad_token
        return states, rewards

    def run(self):
        if self.args.batch_size < 1:
            raise ValueError("batch_size must be positive")
        initilaize_logger()
        model = RewardPlanner(self.args).to(self.args.device)

        try:
            for batch_index in range(self.args.episodes):
                examples = [self.collect_sequence() for _ in range(self.args.batch_size)]
                states = torch.stack([example[0] for example in examples]).to(self.args.device)
                rewards = torch.stack([example[1] for example in examples]).to(self.args.device)
                successful_count = (rewards == RewardPlanner.REWARD_TOKEN).any(dim=1).sum().item()

                losses = [model.train_batch(states, rewards) for _ in range(self.args.epochs)]
                log(
                    f"Planner batch {batch_index + 1}: loss={sum(losses) / len(losses):.4f}, "
                    f"rewarded sequences={successful_count}/{self.args.batch_size}"
                )

            os.makedirs("res", exist_ok=True)
            torch.save(model.state_dict(), "res/reward_planner.pth")
            log("Saved reward planner to res/reward_planner.pth")

            start_states = torch.tensor([0], dtype=torch.long, device=self.args.device)
            generated_states, generated_rewards = model.generate(start_states)
            log(f"Planner states: {generated_states}")
            log(f"Planner rewards: {generated_rewards}")
        finally:
            self.env.close()


if __name__ == "__main__":
    RewardPlannerTraining(parse_args()).run()
