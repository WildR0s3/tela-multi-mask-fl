from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReplayEntry:
    actions: tuple[int, ...]
    states: tuple[int, ...]
    score: float


class CuriosityReplayBuffer:
    """Keep the highest-loss unique sequences under a fixed seven-token mask."""

    def __init__(self, capacity: int = 16, mask_start: int = 2, mask_count: int = 7):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.mask_start = mask_start
        self.mask_count = mask_count
        self.entries: list[ReplayEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _model_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    @torch.no_grad()
    def score_sequences(
        self,
        model   : torch.nn.Module,
        actions : torch.Tensor,
        states  : torch.Tensor,     ) -> list[float]:
        """Score each row by mean CE on valid action/state targets at mask positions 2..8."""
        device = self._model_device(model)
        actions = actions.to(device=device, dtype=torch.long)
        states = states.to(device=device, dtype=torch.long)
        batch_size = states.shape[0]
        args = model.args
        valid = model.build_valid_token_mask(states, args)

        fixed_mask = torch.zeros((batch_size, args.seq_length), dtype=torch.bool, device=device)
        mask_end = self.mask_start + self.mask_count
        if self.mask_start < 0 or mask_end > args.seq_length:
            raise ValueError("fixed mask range exceeds configured sequence length")
        fixed_mask[:, self.mask_start:mask_end] = True

        was_training = model.training
        model.eval()
        try:
            logits_states, logits_actions, mask, _, targets = model(
                actions, states, valid, fixed_mask=fixed_mask
            )

            repeats = model.repeats
            state_indices = torch.arange(3, args.seq_length - 1, 2, device=device)
            action_indices = torch.arange(2, args.seq_length - 1, 2, device=device)
            targets_states = targets[:, state_indices].repeat_interleave(repeats, dim=0)
            targets_actions = targets[:, action_indices].repeat_interleave(repeats, dim=0)
            valid_aug = valid.repeat_interleave(repeats, dim=0)
            state_mask = mask[:, state_indices].bool() & valid_aug[:, state_indices]
            action_mask = mask[:, action_indices].bool() & valid_aug[:, action_indices]

            state_losses = F.cross_entropy(
                logits_states.reshape(-1, args.state_dim),
                targets_states.reshape(-1),
                reduction="none",
            ).reshape(batch_size, repeats, -1)
            action_losses = F.cross_entropy(
                logits_actions.reshape(-1, args.action_dim),
                targets_actions.reshape(-1),
                reduction="none",
            ).reshape(batch_size, repeats, -1)
            state_mask = state_mask.reshape(batch_size, repeats, -1)
            action_mask = action_mask.reshape(batch_size, repeats, -1)

            state_counts = state_mask.sum(dim=(1, 2))
            action_counts = action_mask.sum(dim=(1, 2))
            state_means = (state_losses * state_mask).sum(dim=(1, 2)) / state_counts.clamp(min=1)
            action_means = (action_losses * action_mask).sum(dim=(1, 2)) / action_counts.clamp(min=1)
            available_heads = (state_counts > 0).float() + (action_counts > 0).float()
            scores = (state_means + action_means) / available_heads.clamp(min=1)
            return scores.cpu().tolist()
        finally:
            model.train(was_training)

    @torch.no_grad()
    def update(
        self,
        model   : torch.nn.Module,
        actions : torch.Tensor,
        states  : torch.Tensor,   ) -> None:
        """Rescore incumbents and candidates, then retain the hardest unique examples."""
        candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], None] = {}
        for entry in self.entries:
            candidates[(entry.actions, entry.states)] = None

        actions_cpu = actions.detach().to(device="cpu", dtype=torch.long)
        states_cpu = states.detach().to(device="cpu", dtype=torch.long)
        for action_row, state_row in zip(actions_cpu, states_cpu):
            action_key = tuple(action_row.tolist())
            state_key = tuple(state_row.tolist())
            candidates[(action_key, state_key)] = None

        if not candidates:
            return

        action_rows = [key[0] for key in candidates]
        state_rows = [key[1] for key in candidates]
        device = self._model_device(model)
        candidate_actions = torch.tensor(action_rows, dtype=torch.long, device=device)
        candidate_states = torch.tensor(state_rows, dtype=torch.long, device=device)
        scores = self.score_sequences(model, candidate_actions, candidate_states)

        rescored = [
            ReplayEntry(action_key, state_key, score)
            for (action_key, state_key), score in zip(candidates, scores)
        ]
        self.entries = sorted(rescored, key=lambda entry: entry.score, reverse=True)[: self.capacity]

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample replay rows; sample with replacement during buffer warm-up if needed."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.entries:
            raise ValueError("cannot sample from an empty replay buffer")

        if len(self.entries) >= batch_size:
            selected = random.sample(self.entries, batch_size)
        else:
            selected = [random.choice(self.entries) for _ in range(batch_size)]

        actions = torch.tensor([entry.actions for entry in selected], dtype=torch.long, device=device)
        states = torch.tensor([entry.states for entry in selected], dtype=torch.long, device=device)
        return actions, states
