from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class PlannerReplayEntry:
    tokens: tuple[int, ...]
    score: float
    successful: bool


class PlannerReplayBuffer:
    """Keep unique planner sequences, prioritizing successful sequences."""

    def __init__(self, capacity: int = 32, max_successful_entries: int = 16):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if max_successful_entries < 1:
            raise ValueError("max_successful_entries must be positive")
        self.capacity = capacity
        self.max_successful_entries = min(max_successful_entries, capacity)
        self.entries: list[PlannerReplayEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def unique_count(self) -> int:
        return len({entry.tokens for entry in self.entries})

    @property
    def best_score(self) -> float | None:
        return max((entry.score for entry in self.entries), default=None)

    @property
    def successful_count(self) -> int:
        return sum(entry.successful for entry in self.entries)

    @torch.no_grad()
    def update(self, model, tokens: torch.Tensor) -> None:
        candidates = {entry.tokens for entry in self.entries}

        tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.long)
        for token_row in tokens_cpu:
            candidates.add(tuple(token_row.tolist()))

        if not candidates:
            return

        device = next(model.parameters()).device
        candidate_keys = list(candidates)
        candidate_tokens = torch.tensor(candidate_keys, dtype=torch.long, device=device)
        scores = model.score_sequences(candidate_tokens)
        reward_positions = model.REWARD_POSITIONS
        rewarded_entries = []
        unrewarded_entries = []
        for token_key, score in zip(candidate_keys, scores):
            successful = any(
                token_key[position] == model.REWARD_TOKEN
                for position in reward_positions
            )
            entry = PlannerReplayEntry(token_key, score, successful)
            (rewarded_entries if successful else unrewarded_entries).append(entry)

        rewarded_entries.sort(key=lambda entry: entry.score, reverse=True)
        unrewarded_entries.sort(key=lambda entry: entry.score, reverse=True)
        prioritized_entries = rewarded_entries[: self.max_successful_entries]
        remaining_capacity = self.capacity - len(prioritized_entries)
        prioritized_entries.extend(unrewarded_entries[:remaining_capacity])
        self.entries = prioritized_entries

    def sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.entries:
            raise ValueError("cannot sample from an empty planner replay buffer")

        successful_entries = [entry for entry in self.entries if entry.successful]
        unrewarded_entries = [entry for entry in self.entries if not entry.successful]
        selected = random.sample(successful_entries, min(batch_size, len(successful_entries)))
        remaining_count = batch_size - len(selected)
        selected.extend(
            random.sample(unrewarded_entries, min(remaining_count, len(unrewarded_entries)))
        )
        if len(selected) < batch_size:
            available_entries = selected.copy()
            selected.extend(
                available_entries[index % len(available_entries)]
                for index in range(batch_size - len(available_entries))
            )
        return torch.tensor([entry.tokens for entry in selected], dtype=torch.long, device=device)

    def prepare_batch(
        self,
        model,
        fresh_tokens: torch.Tensor,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Update from fresh rows, then sample with successful rows taking priority."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.update(model, fresh_tokens)
        if not self.entries:
            return torch.empty(
                (0, fresh_tokens.shape[1]), dtype=torch.long, device=device
            )
        return self.sample(batch_size, device)
