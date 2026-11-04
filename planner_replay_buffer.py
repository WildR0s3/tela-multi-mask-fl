from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class PlannerReplayEntry:
    tokens: tuple[int, ...]
    score: float


class PlannerReplayBuffer:
    """Retain planner sequences with the highest fixed-mask loss."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.entries: list[PlannerReplayEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    @torch.no_grad()
    def update(self, model, tokens: torch.Tensor) -> None:
        candidates: dict[tuple[int, ...], None] = {}
        for entry in self.entries:
            candidates[entry.tokens] = None

        tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.long)
        for token_row in tokens_cpu:
            candidates[tuple(token_row.tolist())] = None

        if not candidates:
            return

        device = next(model.parameters()).device
        candidate_tokens = torch.tensor(list(candidates), dtype=torch.long, device=device)
        scores = model.score_sequences(candidate_tokens)
        self.entries = sorted(
            (
                PlannerReplayEntry(token_key, score)
                for token_key, score in zip(candidates, scores)
            ),
            key=lambda entry: entry.score,
            reverse=True,
        )[: self.capacity]

    def sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.entries:
            raise ValueError("cannot sample from an empty planner replay buffer")
        selected = (
            random.sample(self.entries, batch_size)
            if len(self.entries) >= batch_size
            else [random.choice(self.entries) for _ in range(batch_size)]
        )
        return torch.tensor([entry.tokens for entry in selected], dtype=torch.long, device=device)
