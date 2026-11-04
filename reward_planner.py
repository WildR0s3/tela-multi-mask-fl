import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_modules.encoder import EncoderBlock


class RewardPlanner(nn.Module):
    SEQUENCE_LENGTH = 11
    STATE_POSITIONS = (0, 1, 3, 5, 7, 9, 10)
    REWARD_POSITIONS = (2, 4, 6, 8)
    TARGET_POSITIONS = (2, 3, 4, 5, 6, 7, 9)
    TARGET_STATE_POSITIONS = (3, 5, 7, 9)
    TARGET_REWARD_POSITIONS = (2, 4, 6)
    NO_REWARD_TOKEN = 19
    REWARD_TOKEN = 20
    REWARD_VOCAB_SIZE = 2
    REWARD_MASK_INDEX = 21
    MASK_REPEATS = 16

    def __init__(self, args):
        super().__init__()
        if args.state_dim <= args.pad_token:
            raise ValueError("state_dim must include START, TERM, and PAD tokens")

        self.args = args
        if args.state_dim <= args.pad_token:
            raise ValueError("state_dim must include the state PAD token")

        self.state_embedding = nn.Embedding(args.state_dim, args.embedding_dim)
        self.reward_embedding = nn.Embedding(self.REWARD_VOCAB_SIZE, args.embedding_dim)
        self.state_channel_embedding = nn.Parameter(torch.empty(1, 1, args.embedding_dim))
        self.reward_channel_embedding = nn.Parameter(torch.empty(1, 1, args.embedding_dim))
        nn.init.normal_(self.state_channel_embedding, std=0.02)
        nn.init.normal_(self.reward_channel_embedding, std=0.02)
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

    def forward(self, tokens, mask):
        if tokens.shape != mask.shape:
            raise ValueError("tokens and mask must have matching [batch, 11] shapes")
        if tokens.ndim != 2 or tokens.shape[1] != self.SEQUENCE_LENGTH:
            raise ValueError(f"planner sequences must have length {self.SEQUENCE_LENGTH}")

        positions = torch.arange(self.SEQUENCE_LENGTH, device=tokens.device)
        reward_slot = torch.zeros(self.SEQUENCE_LENGTH, dtype=torch.bool, device=tokens.device)
        reward_slot[list(self.REWARD_POSITIONS)] = True
        reward_slot = reward_slot.unsqueeze(0)

        state_input = tokens.masked_fill(reward_slot, 0)
        reward_input = (tokens - self.NO_REWARD_TOKEN).clamp(0, self.REWARD_VOCAB_SIZE - 1)
        state_values = self.state_embedding(state_input) + self.state_channel_embedding
        reward_values = self.reward_embedding(reward_input) + self.reward_channel_embedding
        mask_values = torch.where(
            reward_slot.unsqueeze(-1), self.reward_mask_token, self.state_mask_token
        )
        x = torch.where(reward_slot.unsqueeze(-1), reward_values, state_values)
        x = torch.where(mask.unsqueeze(-1), mask_values, x)
        x = x + self.position_embedding(positions)

        valid_positions = (tokens != self.args.pad_token) | mask
        for block in self.blocks:
            x = block(x, valid_positions)
        x = self.final_norm(x)

        state_logits = self.state_head(x)
        state_logits[..., self.args.start_token] = torch.finfo(state_logits.dtype).min
        state_logits[..., self.args.pad_token] = torch.finfo(state_logits.dtype).min
        reward_logits = self.reward_head(x)
        return state_logits, reward_logits

    @classmethod
    def interleave_sequences(cls, states, rewards, pad_token, term_token):
        if states.ndim != 2 or states.shape[1] != 7 or rewards.shape != states.shape:
            raise ValueError("compressed state and reward batches must both have shape [batch, 7]")
        tokens = torch.full(
            (states.shape[0], cls.SEQUENCE_LENGTH), pad_token, dtype=states.dtype, device=states.device
        )
        tokens[:, 0] = states[:, 0]
        tokens[:, 1] = states[:, 1]
        for edge_index in range(4):
            compressed_position = edge_index + 2
            reward_position = edge_index * 2 + 2
            state_position = reward_position + 1
            ended = states[:, compressed_position] == term_token
            tokens[:, reward_position] = torch.where(
                ended, term_token, rewards[:, compressed_position]
            )
            tokens[:, state_position] = torch.where(
                ended, pad_token, states[:, compressed_position]
            )
        tokens[:, 10] = states[:, 6]
        return tokens

    def _valid_target_mask(self, tokens):
        device = next(self.parameters()).device
        valid = torch.zeros_like(tokens, dtype=torch.bool, device=device)
        state_positions = list(self.TARGET_STATE_POSITIONS)
        reward_positions = list(self.TARGET_REWARD_POSITIONS)
        state_targets = tokens[:, state_positions]
        reward_targets = tokens[:, reward_positions]
        valid[:, state_positions] = (state_targets != self.args.pad_token) & (state_targets != self.args.start_token)
        valid[:, reward_positions] = (reward_targets == self.NO_REWARD_TOKEN) | (reward_targets == self.REWARD_TOKEN)
        return valid

    def _make_training_masks(self, tokens):
        valid_targets = self._valid_target_mask(tokens)
        random_values = torch.rand(valid_targets.shape, device=tokens.device)
        random_values = random_values.masked_fill(~valid_targets, float("inf"))
        random_ranks = random_values.argsort(dim=-1).argsort(dim=-1)
        valid_count = valid_targets.sum(dim=1, keepdim=True)
        mask_sizes = torch.randint(1, 8, (valid_targets.shape[0], 1), device=tokens.device)
        mask_sizes = torch.minimum(mask_sizes, valid_count)
        return (random_ranks < mask_sizes) & valid_targets

    def _loss(self, tokens, mask):
        state_logits, reward_logits = self(tokens, mask)
        state_positions = list(self.TARGET_STATE_POSITIONS)
        reward_positions = list(self.TARGET_REWARD_POSITIONS)
        state_mask = mask[:, state_positions]
        reward_mask = mask[:, reward_positions]
        state_targets = tokens[:, state_positions]
        safe_state_targets = torch.where(state_mask, state_targets, torch.zeros_like(state_targets))
        state_losses = F.cross_entropy(
            state_logits[:, state_positions, :].reshape(-1, self.args.state_dim),
            safe_state_targets.reshape(-1),
            reduction="none",
        ).reshape_as(state_mask)
        reward_targets = (tokens[:, reward_positions] == self.REWARD_TOKEN).long()
        reward_losses = F.cross_entropy(
            reward_logits[:, reward_positions, :].reshape(-1, self.REWARD_VOCAB_SIZE),
            reward_targets.reshape(-1),
            reduction="none",
        ).reshape_as(reward_mask)
        masked_count = state_mask.sum() + reward_mask.sum()
        loss = (
            (state_losses * state_mask).sum() + (reward_losses * reward_mask).sum()
        ) / masked_count.clamp(min=1)
        return loss, state_losses, reward_losses, state_mask, reward_mask

    def learn_model(self, tokens):
        device = next(self.parameters()).device
        tokens = tokens.to(device=device, dtype=torch.long)
        if tokens.ndim != 2 or tokens.shape[1] != self.SEQUENCE_LENGTH:
            raise ValueError(f"tokens must have shape [batch, {self.SEQUENCE_LENGTH}]")

        tokens = tokens.repeat_interleave(self.MASK_REPEATS, dim=0)
        mask = self._make_training_masks(tokens)
        loss, _, _, _, _ = self._loss(tokens, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    @torch.no_grad()
    def score_sequences(self, tokens):
        device = next(self.parameters()).device
        tokens = tokens.to(device=device, dtype=torch.long)
        if tokens.ndim != 2 or tokens.shape[1] != self.SEQUENCE_LENGTH:
            raise ValueError("tokens must have shape [batch, 11]")
        mask = self._valid_target_mask(tokens)

        was_training = self.training
        self.eval()
        try:
            _, state_losses, reward_losses, state_mask, reward_mask = self._loss(tokens, mask)
            state_count = state_mask.sum(dim=1)
            reward_count = reward_mask.sum(dim=1)
            total_count = state_count + reward_count
            scores = (
                (state_losses * state_mask).sum(dim=1)
                + (reward_losses * reward_mask).sum(dim=1)
            ) / total_count.clamp(min=1)
            return scores.cpu().tolist()
        finally:
            self.train(was_training)

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

        tokens = torch.full(
            (batch_size, self.SEQUENCE_LENGTH), self.args.pad_token, dtype=torch.long, device=device
        )
        tokens[:, 0] = self.args.start_token
        tokens[:, 1] = start_states
        tokens[:, 8] = self.REWARD_TOKEN
        tokens[:, 10] = self.args.term_token
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[:, list(self.TARGET_POSITIONS)] = True

        was_training = self.training
        self.eval()
        try:
            for tokens_to_unmask in schedule:
                state_logits, reward_logits = self(tokens, mask)
                state_confidence = self._confidence(state_logits)
                reward_confidence = self._confidence(reward_logits)

                for batch_index in range(batch_size):
                    candidates = []
                    for position in self.TARGET_STATE_POSITIONS:
                        if mask[batch_index, position]:
                            candidates.append((
                                state_confidence[batch_index, position].item(),
                                position,
                                state_logits[batch_index, position].argmax(),
                            ))
                    for position in self.TARGET_REWARD_POSITIONS:
                        if mask[batch_index, position]:
                            candidates.append((
                                reward_confidence[batch_index, position].item(),
                                position,
                                reward_logits[batch_index, position].argmax() + self.NO_REWARD_TOKEN,
                            ))
                    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:tokens_to_unmask]
                    for _, position, token in selected:
                        tokens[batch_index, position] = token
                        mask[batch_index, position] = False

            self._truncate_after_termination(tokens)
            return tokens
        finally:
            self.train(was_training)

    def _truncate_after_termination(self, tokens):
        for batch_index in range(tokens.shape[0]):
            reward_positions = [
                position for position in self.TARGET_REWARD_POSITIONS
                if tokens[batch_index, position].item() == self.REWARD_TOKEN
            ]
            state_terminal_positions = [
                position for position in self.TARGET_STATE_POSITIONS
                if tokens[batch_index, position].item() == self.args.term_token
            ]
            terminal_positions = state_terminal_positions
            if reward_positions:
                terminal_positions.append(reward_positions[0] + 2)
            if terminal_positions:
                terminal_position = min(terminal_positions)
                if terminal_position < 10:
                    tokens[batch_index, terminal_position] = self.args.term_token
                    tokens[batch_index, terminal_position + 1:] = self.args.pad_token
