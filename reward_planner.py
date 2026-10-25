import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_modules.encoder import EncoderBlock


class RewardPlanner(nn.Module):
    SEQUENCE_LENGTH = 7
    NO_REWARD_TOKEN = 19
    REWARD_TOKEN = 20
    REWARD_VOCAB_SIZE = 21
    MASK_REPEATS = 16

    def __init__(self, args):
        super().__init__()
        if args.state_dim <= args.pad_token:
            raise ValueError("state_dim must include START, TERM, and PAD tokens")

        self.args = args
        self.state_embedding = nn.Embedding(args.state_dim, args.embedding_dim)
        self.reward_embedding = nn.Embedding(self.REWARD_VOCAB_SIZE, args.embedding_dim)
        self.position_embedding = nn.Embedding(self.SEQUENCE_LENGTH, args.embedding_dim)
        self.state_mask_token = nn.Parameter(torch.empty(1, 1, args.embedding_dim))
        self.reward_mask_token = nn.Parameter(torch.empty(1, 1, args.embedding_dim))
        nn.init.normal_(self.state_mask_token, std=0.02)
        nn.init.normal_(self.reward_mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            EncoderBlock(args.embedding_dim, args.num_heads, args.dropout)
            for _ in range(args.num_layers)
        )
        self.final_norm = nn.LayerNorm(args.embedding_dim)
        self.state_head = nn.Linear(args.embedding_dim, args.state_dim)
        self.reward_head = nn.Linear(args.embedding_dim, self.REWARD_VOCAB_SIZE)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=args.learning_rate)

    def forward(self, states, rewards, state_mask, reward_mask):
        if states.shape != rewards.shape or states.shape != state_mask.shape or states.shape != reward_mask.shape:
            raise ValueError("states, rewards, and masks must have matching [batch, 7] shapes")
        if states.shape[1] != self.SEQUENCE_LENGTH:
            raise ValueError(f"planner sequences must have length {self.SEQUENCE_LENGTH}")

        state_values = self.state_embedding(states)
        reward_values = self.reward_embedding(rewards)
        state_values = torch.where(state_mask.unsqueeze(-1), self.state_mask_token, state_values)
        reward_values = torch.where(reward_mask.unsqueeze(-1), self.reward_mask_token, reward_values)
        positions = torch.arange(self.SEQUENCE_LENGTH, device=states.device)
        x = state_values + reward_values + self.position_embedding(positions)

        valid_positions = (
            (states != self.args.pad_token)
            | (rewards != self.args.pad_token)
            | state_mask
            | reward_mask
        )
        for block in self.blocks:
            x = block(x, valid_positions)
        x = self.final_norm(x)

        state_logits = self.state_head(x)
        state_logits[..., self.args.start_token] = torch.finfo(state_logits.dtype).min
        reward_logits = self.reward_head(x)
        reward_logits[..., : self.args.start_token + 1] = torch.finfo(reward_logits.dtype).min
        return state_logits, reward_logits

    def train_batch(self, states, rewards):
        device = next(self.parameters()).device
        states = states.to(device=device, dtype=torch.long)
        rewards = rewards.to(device=device, dtype=torch.long)
        if states.ndim != 2 or states.shape[1] != self.SEQUENCE_LENGTH:
            raise ValueError(f"states must have shape [batch, {self.SEQUENCE_LENGTH}]")
        if rewards.shape != states.shape:
            raise ValueError("reward sequences must match the state sequence shape")

        targets_states = states.repeat_interleave(self.MASK_REPEATS, dim=0)
        targets_rewards = rewards.repeat_interleave(self.MASK_REPEATS, dim=0)
        input_states = targets_states.clone()
        input_rewards = targets_rewards.clone()
        # Condition every training example on the same eventual-reward goal token.
        input_rewards[:, 5] = self.REWARD_TOKEN

        state_targets_valid = targets_states[:, 2:6] != self.args.pad_token
        reward_targets = targets_rewards[:, 2:5]
        reward_targets_valid = (reward_targets == self.NO_REWARD_TOKEN) | (reward_targets == self.REWARD_TOKEN)
        valid_targets = torch.cat((state_targets_valid, reward_targets_valid), dim=1)
        random_values = torch.rand(valid_targets.shape, device=device)
        random_values = random_values.masked_fill(~valid_targets, float("inf"))
        random_ranks = random_values.argsort(dim=-1).argsort(dim=-1)
        valid_count = valid_targets.sum(dim=1, keepdim=True)
        mask_sizes = torch.randint(1, 8, (valid_targets.shape[0], 1), device=device)
        mask_sizes = torch.minimum(mask_sizes, valid_count)
        chosen = (random_ranks < mask_sizes) & valid_targets

        state_mask = torch.zeros_like(input_states, dtype=torch.bool)
        reward_mask = torch.zeros_like(input_rewards, dtype=torch.bool)
        state_mask[:, 2:6] = chosen[:, :4]
        reward_mask[:, 2:5] = chosen[:, 4:]

        state_logits, reward_logits = self(input_states, input_rewards, state_mask, reward_mask)
        state_losses = F.cross_entropy(
            state_logits.reshape(-1, self.args.state_dim),
            targets_states.reshape(-1),
            reduction="none",
        ).reshape_as(state_mask)
        reward_losses = F.cross_entropy(
            reward_logits.reshape(-1, self.REWARD_VOCAB_SIZE),
            targets_rewards.reshape(-1),
            reduction="none",
        ).reshape_as(reward_mask)
        total_masked = state_mask.sum() + reward_mask.sum()
        loss = (
            (state_losses * state_mask).sum() + (reward_losses * reward_mask).sum()
        ) / total_masked.clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    @staticmethod
    def _confidence(logits):
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        valid_class_count = (logits > torch.finfo(logits.dtype).min / 2).sum(dim=-1).clamp(min=2)
        return 1.0 - entropy / valid_class_count.float().log()

    @torch.no_grad()
    def generate(self, start_states, schedule=(3, 2, 2)):
        if tuple(schedule) != (3, 2, 2):
            raise ValueError("planner generation currently uses the fixed 3,2,2 schedule")
        device = next(self.parameters()).device
        start_states = torch.as_tensor(start_states, dtype=torch.long, device=device).reshape(-1)
        batch_size = start_states.shape[0]

        states = torch.full(
            (batch_size, self.SEQUENCE_LENGTH), self.args.pad_token, dtype=torch.long, device=device
        )
        rewards = torch.full_like(states, self.args.pad_token)
        states[:, 0] = self.args.start_token
        states[:, 1] = start_states
        states[:, 6] = self.args.term_token
        rewards[:, 0] = self.args.start_token
        rewards[:, 1] = self.NO_REWARD_TOKEN
        rewards[:, 5] = self.REWARD_TOKEN

        state_mask = torch.zeros_like(states, dtype=torch.bool)
        reward_mask = torch.zeros_like(rewards, dtype=torch.bool)
        state_mask[:, 2:6] = True
        reward_mask[:, 2:5] = True

        was_training = self.training
        self.eval()
        try:
            for tokens_to_unmask in schedule:
                state_logits, reward_logits = self(states, rewards, state_mask, reward_mask)
                state_confidence = self._confidence(state_logits)
                reward_confidence = self._confidence(reward_logits)

                for batch_index in range(batch_size):
                    candidates = []
                    for position in range(2, 6):
                        if state_mask[batch_index, position]:
                            candidates.append((
                                state_confidence[batch_index, position].item(),
                                "state",
                                position,
                                state_logits[batch_index, position].argmax(),
                            ))
                    for position in range(2, 5):
                        if reward_mask[batch_index, position]:
                            candidates.append((
                                reward_confidence[batch_index, position].item(),
                                "reward",
                                position,
                                reward_logits[batch_index, position].argmax(),
                            ))
                    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:tokens_to_unmask]
                    for _, stream, position, token in selected:
                        if stream == "state":
                            states[batch_index, position] = token
                            state_mask[batch_index, position] = False
                        else:
                            rewards[batch_index, position] = token
                            reward_mask[batch_index, position] = False

            self._truncate_after_reward(states, rewards)
            return states, rewards
        finally:
            self.train(was_training)

    def _truncate_after_reward(self, states, rewards):
        for batch_index in range(states.shape[0]):
            reward_positions = [
                position for position in range(2, 5)
                if rewards[batch_index, position].item() == self.REWARD_TOKEN
            ]
            if reward_positions:
                terminal_position = reward_positions[0] + 1
                if terminal_position < 6:
                    states[batch_index, terminal_position] = self.args.term_token
                    rewards[batch_index, terminal_position] = self.args.term_token
                    states[batch_index, terminal_position + 1:] = self.args.pad_token
                    rewards[batch_index, terminal_position + 1:] = self.args.pad_token
                continue

            early_stop_positions = [
                position for position in range(2, 5)
                if states[batch_index, position].item() in (self.args.term_token, self.args.pad_token)
                or rewards[batch_index, position].item() in (self.args.term_token, self.args.pad_token)
            ]
            if early_stop_positions:
                terminal_position = early_stop_positions[0]
                states[batch_index, terminal_position] = self.args.term_token
                rewards[batch_index, terminal_position] = self.args.term_token
                states[batch_index, terminal_position + 1:] = self.args.pad_token
                rewards[batch_index, terminal_position + 1:] = self.args.pad_token
