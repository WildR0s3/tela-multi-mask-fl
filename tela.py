
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_modules.encoder import EncoderBlock
from logger.logging_module import log, lte
from masks import Masks
import copy, math
from collections import defaultdict

# TODO - continue with testing random masks with cosine or exp schedule
#       process_and_augment_batch seems to be right for now.
# TODO - divide your sequence of 256 in 4 batches of 64, 16 from each separate sequence then 
# maybe i want to apply 512 masks on each sequence giving me whoooping (4 sequences x 512 = 2048) examples 
# and then i can iterate over that in batches of 64 (16 from each sequence) and then i will have 32 epochs to run on
######## APPLY THE MASK #########
# TODO - I need to make this traning happen and see the results. 
# TODO - I need to make MaskGIT like sequence generation for later stages. (so generating all 7 states in 3 steps)
# TODO - Keep in mind T-ELA Explore, Learn, Act
# TODO - I need probably another classification token that will be the goal and summary from higher level transformer and to higher level transformer
#        (should it act both ways? in like exchange of token information? token from lower layer, adn goal from higher layer)

# EMBEDING_DIM    = 128
# ACTION_DIM      = 4
# STATE_DIM       = 16 + 3 # states + TERM + START, PAD will be masked sbut let's add it here to allow full batch embedding alogn with tid
# DROP_OUT        = 0.0
# NUM_OF_HEADS    = 16
# NUM_OF_LAYERS   = 6
# LEARNING_RATE   = 1e-3
# SEQ_LENGTH      = 11
# DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE      = 4
# START_TOKEN     = 16
# TERM_TOKEN      = 17
# PAD_TOKEN       = 18
torch.set_printoptions(precision=4, sci_mode=False)


@dataclass
class TrainingConfig:
    embedding_dim   : int 
    action_dim      : int 
    state_dim       : int 
    drop_out        : float 
    num_of_heads    : int 
    num_of_layers   : int
    learning_rate   : float 
    seq_length      : int 
    device          : str
    batch_size      : int
    start_token     : int
    term_token      : int
    pad_token       : int 


class TELA(nn.Module):

    def __init__(self, args : TrainingConfig):
        super().__init__()
        self.args = args
        self.states_embed = nn.Embedding(self.args.state_dim, self.args.embedding_dim)
        self.action_embed = nn.Embedding(self.args.action_dim, self.args.embedding_dim)
        self.mask_strategy = 'exp'

        self.mask_action_token = nn.Parameter(torch.zeros(1, 1, self.args.embedding_dim))
        nn.init.normal_(self.mask_action_token, std=0.02)
        self.mask_state_token = nn.Parameter(torch.zeros(1, 1, self.args.embedding_dim))
        nn.init.normal_(self.mask_state_token, std=0.02)

        self.pos_embedding = nn.Embedding(self.args.seq_length, self.args.embedding_dim)
        self.action_type_embed = nn.Parameter(torch.randn(1, 1, self.args.embedding_dim))
        nn.init.normal_(self.action_type_embed, std=0.02)
        self.state_type_embed = nn.Parameter(torch.randn(1, 1, self.args.embedding_dim))
        nn.init.normal_(self.state_type_embed, std=0.02)

        self.blocks = nn.ModuleList([EncoderBlock(self.args.embedding_dim, self.args.num_heads, self.args.drop_out) for _ in range(self.args.num_layers)])

        self.ln_f = nn.LayerNorm(self.args.embedding_dim)
        self.action_head = nn.Linear(self.args.embedding_dim, self.args.action_dim)
        self.states_head = nn.Linear(self.args.embedding_dim, self.args.state_dim)

        self.criterion = nn.CrossEntropyLoss()
        self.model_optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate) # weight_decay=1e-4 // for a small model remove

        m = Masks()
        m.create_masks(self.args.seq_length)
        self.masks = m.masks_dict
        self.repeats = 16



    @staticmethod
    def build_valid_token_mask(states, args: TrainingConfig):
        """Mark real tokens; TERM is a state target but not an executed action."""
        batch_size = states.shape[0]
        valid = torch.zeros((batch_size, args.seq_length), dtype=torch.bool, device=states.device)

        valid[:, 0] = states[:, 0] != args.pad_token
        valid[:, 1] = states[:, 1] != args.pad_token
        num_steps = (args.seq_length - 3) // 2
        for step in range(num_steps):
            action_pos = 2 + step * 2
            state_pos = action_pos + 1
            previous_state = states[:, step + 1]
            previous_state_valid = (previous_state != args.pad_token) & (previous_state != args.term_token)
            next_state_valid = states[:, step + 2] != args.pad_token
            action_next_state_valid = states[:, step + 2] != args.term_token
            valid[:, action_pos] = previous_state_valid & next_state_valid & action_next_state_valid
            valid[:, state_pos] = next_state_valid

        valid[:, 10] = states[:, 6] != args.pad_token
        return valid


    def forward(self, actions, states, valid_token_mask, fixed_mask=None):

        states_emb = self.states_embed(states)
        actions_emb = self.action_embed(actions)
        # for i in range(BATCH_SIZE // 8):
        batch_size = states.shape[0]
        sequence = torch.zeros((batch_size, self.args.seq_length, self.args.embedding_dim), device=states.device)

        targets = torch.zeros((batch_size, self.args.seq_length), dtype=torch.long, device=states.device)
        targets[:, :2] = states[:, :2]
        targets[:, 2:10:2] = actions
        targets[:, 3::2] = states[:, 2:6]
        targets[:, -1] = states[:, -1]

        sequence[:, 0] = states_emb[:, 0, :]
        sequence[:, 1] = states_emb[:, 1, :]

        num_steps = (self.args.seq_length - 3) // 2

        action_indices = torch.arange(2, 2 + num_steps * 2, step=2, device=self.args.device)
        state_indices  = torch.arange(3, 3 + num_steps * 2, step=2, device=self.args.device)

        sequence[:, action_indices] = actions_emb[:, :num_steps, :]
        sequence[:, state_indices]  = states_emb[:, 2:2+num_steps, :]

        sequence[:, 10] = states_emb[:, 6, :]

        x_aug, mask, p_mask = self.process_and_augment_batch(sequence, valid_token_mask, fixed_mask)

        indices = torch.arange(self.args.seq_length, device=states.device)
        is_even = ((indices % 2 == 0).unsqueeze(0).unsqueeze(-1))
        to_be_masked = torch.where(is_even, self.mask_action_token, self.mask_state_token)

        # TODO OLD - it needs to be applied, but possibly on merged IDs for the padding_mask is_sepcial legnth matching
        special_token_ids = [16, 17, 18]
        is_special_states = torch.isin(states, torch.tensor(special_token_ids, device=states.device))
        is_special_seq = torch.zeros((batch_size, self.args.seq_length), dtype=torch.bool, device=states.device)
        state_seq_indices = torch.tensor([0, 1, 3, 5, 7, 9, 10], device=states.device)
        is_special_seq[:, state_seq_indices] = is_special_states
        is_data = valid_token_mask.bool() & (~is_special_seq)

        # TODO OLD - to be verified, especially the size if troch arrange is needed
        is_even_expanded = is_even.expand(is_data.shape[0], -1, -1)
        type_embed = torch.where(is_even_expanded, self.action_type_embed, self.state_type_embed) 
        types_embed_adjusted = torch.where(is_data.unsqueeze(-1), type_embed, 0.)
        types_embed_adjusted = types_embed_adjusted.repeat_interleave(self.repeats, dim=0)

        mask_expanded = mask.unsqueeze(-1)
        x_masked = torch.where(mask_expanded == 1.0, to_be_masked, x_aug)
        position_embed = self.pos_embedding(torch.arange(self.args.seq_length, device=states.device))
        # TODO OLD - to be verified addign type embed
        x = x_masked + position_embed + types_embed_adjusted 
        padding_mask_aug = valid_token_mask.repeat_interleave(self.repeats, dim=0)
        for block in self.blocks:
            x = block(x, padding_mask_aug)
        x = self.ln_f(x)

        x_states = x[:, state_indices, :]   # Shape: (BATCH_SIZE, 4, EMBEDDING_DIM)
        x_actions = x[:, action_indices, :] # Shape: (BATCH_SIZE, 4, EMBEDDING_DIM)

        logits_states = self.states_head(x_states)   # Shape: (BATCH_SIZE, 4, STATE_DIM)
        logits_actions = self.action_head(x_actions) # Shape: (BATCH_SIZE, 4, ACTION_DIM)
        # TODO: assemble sequence back together then compute losses for actions and states and combine them back together
        return logits_states, logits_actions, mask, p_mask, targets



    def forward_inference(self, actions, states, mask, padding_mask):
        """
        Corrected inference pass matching:
        states sequence: [start_token, s0, s1, s2, s3, s4, term_token]
        """
        batch_size = states.shape[0]
        
        # 1. Embed states and actions
        states_emb = self.states_embed(states)   # (BATCH_SIZE, 7, EMBEDDING_DIM)
        actions_emb = self.action_embed(actions) # (BATCH_SIZE, 4, EMBEDDING_DIM)
        
        # 2. Build sequence representation
        sequence = torch.zeros((batch_size, self.args.seq_length, self.args.embedding_dim), device=self.args.device)
        
        # Place static boundary tokens
        sequence[:, 0] = states_emb[:, 0, :] # start_sequence_token
        sequence[:, 1] = states_emb[:, 1, :] # S0
        sequence[:, 9] = states_emb[:, 5, :] # S4 (Index 5 in states)
        sequence[:, 10] = states_emb[:, 6, :] # term_token (Index 6 in states)
        
        num_steps = (self.args.seq_length - 3) // 2 # 4 steps
        action_indices = torch.arange(2, 2 + num_steps * 2, step=2, device=self.args.device) # [2, 4, 6, 8]
        state_indices  = torch.arange(3, 3 + num_steps * 2, step=2, device=self.args.device) # [3, 5, 7, 9]
        
        # Interleave actions and middle states (S1, S2, S3, S4)
        sequence[:, action_indices] = actions_emb[:, :num_steps, :]
        sequence[:, state_indices]  = states_emb[:, 2:2+num_steps, :] # States S1 to S4 (indices 2,3,4,5)
        
        # 3. Inject [MASK] embeddings
        indices = torch.arange(self.args.seq_length, device=self.args.device)
        is_even = ((indices % 2 == 0).unsqueeze(0).unsqueeze(-1))
        
        to_be_masked = torch.where(is_even, self.mask_action_token, self.mask_state_token)
        mask_expanded = mask.unsqueeze(-1)
        sequence_masked = torch.where(mask_expanded, to_be_masked, sequence)
        
        # 4. Add positional and type embeddings
        type_embed = torch.where(is_even, self.action_type_embed, self.state_type_embed)
        position_embed = self.pos_embedding(indices)
        
        x = sequence_masked + position_embed + type_embed
        
        # 5. Transformer backbone pass
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.ln_f(x)
        
        # 6. Extract target representations and compute logits
        x_states = x[:, state_indices, :]   # (BATCH_SIZE, 4, EMBEDDING_DIM) -> predicts S1, S2, S3, S4
        x_actions = x[:, action_indices, :] # (BATCH_SIZE, 4, EMBEDDING_DIM) -> predicts A0, A1, A2, A3
        
        logits_states = self.states_head(x_states)
        logits_actions = self.action_head(x_actions)
        
        return logits_states, logits_actions

    # Why it worked without 'self'?
    @torch.no_grad()
    def generate_emigm_3steps(self, model, s0, s4, padding_mask):
        """
        Sequence structure:
        Sequence: [start_seq, S0, A0, S1, A1, S2, A2, S3, A3, S4, term_token]
        Seq idx:  [    0,     1,  2,  3,  4,  5,  6,  7,  8,  9,     10   ]
        
        Targets to predict: [A0, S1, A1, S2, A2, S3, A3] -> Seq indices [2, 3, 4, 5, 6, 7, 8]
        """
        model.eval()
        batch_size = s0.shape[0]
        
        # Sequence indices for the 7 predicted tokens
        gen_seq_indices = torch.tensor([2, 3, 4, 5, 6, 7, 8], device=self.args.device)
        action_slot_mask = torch.tensor([True, False, True, False, True, False, True], device=self.args.device)
        
        # Initialize state and action tensors
        states = torch.zeros((batch_size, 7), dtype=torch.long, device=self.args.device)
        actions = torch.zeros((batch_size, 4), dtype=torch.long, device=self.args.device)
        # states.fill_(18) # use pad token as placeholder
        # actions.fill_(18) # use pad token as placeholder
        
        # Assign known boundary state tokens
        states[:, 1] = s0 # S0 is at index 1
        states[:, 5] = s4 # S4 is at index 5
        states[:, 0] = self.args.start_token # start_seq token
        states[:, 6] = self.args.term_token
        
        # Initialize mask tensor (True = MASKED)
        seq_mask = torch.zeros((batch_size, self.args.seq_length), dtype=torch.bool, device=self.args.device)
        seq_mask[:, gen_seq_indices] = True
        
        is_slot_masked = torch.ones((batch_size, 7), dtype=torch.bool, device=self.args.device)
        unmask_schedule = [3, 2, 2]
        
        log_v_state = torch.log(torch.tensor(model.states_head.out_features, dtype=torch.float32, device=self.args.device))
        log_v_action = torch.log(torch.tensor(model.action_head.out_features, dtype=torch.float32, device=self.args.device))

        for num_to_unmask in unmask_schedule:
            logits_states, logits_actions = model.forward_inference(actions, states, seq_mask, padding_mask)
            
            p_states = F.softmax(logits_states, dim=-1)   # Predicts S1, S2, S3, S4
            p_actions = F.softmax(logits_actions, dim=-1) # Predicts A0, A1, A2, A3
            
            ent_states = -torch.sum(p_states * torch.log(p_states + 1e-9), dim=-1) / log_v_state
            ent_actions = -torch.sum(p_actions * torch.log(p_actions + 1e-9), dim=-1) / log_v_action
            
            conf_states = 1.0 - ent_states
            conf_actions = 1.0 - ent_actions
            
            slot_confidences = torch.zeros((batch_size, 7), device=self.args.device)
            slot_predictions = torch.zeros((batch_size, 7), dtype=torch.long, device=self.args.device)
            
            # Actions: Slots 0, 2, 4, 6 -> A0, A1, A2, A3
            slot_confidences[:, 0::2] = conf_actions
            slot_predictions[:, 0::2] = torch.argmax(p_actions, dim=-1)
            
            # States: Slots 1, 3, 5 -> S1, S2, S3 (indices 0, 1, 2 from states_head)
            slot_confidences[:, 1::2] = conf_states[:, :3]
            slot_predictions[:, 1::2] = torch.argmax(p_states[:, :3], dim=-1)
            
            # Ignore already unmasked tokens
            slot_confidences = torch.where(is_slot_masked, slot_confidences, torch.tensor(-1e9, device=self.args.device))
            
            # Pick top-k highest confidence tokens per batch
            _, topk_slot_indices = torch.topk(slot_confidences, k=num_to_unmask, dim=-1)
            
            for b in range(batch_size):
                selected_slots = topk_slot_indices[b]
                for slot_idx in selected_slots:
                    is_slot_masked[b, slot_idx] = False
                    
                    seq_pos = gen_seq_indices[slot_idx]
                    seq_mask[b, seq_pos] = False
                    
                    pred_token = slot_predictions[b, slot_idx]
                    if action_slot_mask[slot_idx]:
                        action_idx = slot_idx // 2
                        actions[b, action_idx] = pred_token
                    else:
                        # Maps slots [1, 3, 5] -> state indices [2, 3, 4] for S1, S2, S3
                        state_idx = (slot_idx // 2) + 2
                        states[b, state_idx] = pred_token

        return actions, states

    # There are 2 alternatives - below is flattening, or i I can use ignore index in F.cross_entropy
    def learn_model(self, logits_states, logits_actions, targets, mask, valid_token_mask, i):

        num_steps = (self.args.seq_length - 3) // 2

        action_indices = torch.arange(2, 2 + num_steps * 2, step=2, device=logits_states.device)
        state_indices  = torch.arange(3, 3 + num_steps * 2, step=2, device=logits_states.device)

        ## TODO: probably to do it like this one 2026.09.30
        # Extract target indices corresponding to sequence positions
        targets_states = targets[:, state_indices].long()
        targets_states_aug = targets_states.repeat_interleave(self.repeats, dim=0)   # Shape: (BATCH_SIZE, 7)
        targets_actions = targets[:, action_indices].long() # Shape: (BATCH_SIZE, 4)
        targets_actions_aug = targets_actions.repeat_interleave(self.repeats, dim=0)

        valid_aug = valid_token_mask.repeat_interleave(self.repeats, dim=0)
        mask_states = mask[:, state_indices].float() * valid_aug[:, state_indices].float()
        mask_actions = mask[:, action_indices].float() * valid_aug[:, action_indices].float()

        # Compute standard cross-entropy per head
        loss_states = F.cross_entropy(
            logits_states.reshape(-1, self.args.state_dim),
            targets_states_aug.reshape(-1),
            reduction="none"
        ) * mask_states.reshape(-1)

        loss_actions = F.cross_entropy(
            logits_actions.reshape(-1, self.args.action_dim),
            targets_actions_aug.reshape(-1),
            reduction="none"
        ) * mask_actions.reshape(-1)

        # Combined loss
        total_masked_tokens = mask_states.sum() + mask_actions.sum()
        loss = (loss_states.sum() + loss_actions.sum()) / total_masked_tokens.clamp(min=1.0)
        if i % 8 == 0:
            log(f"Loss: {loss}")
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()


    @torch.no_grad()
    def test_loss(self, actions, states, fixed_mask):
        """Evaluate fixed examples with a fixed mask, without changing weights."""
        valid_token_mask = self.build_valid_token_mask(states, self.args)
        was_training = self.training
        self.eval()
        try:
            logits_states, logits_actions, mask, _, targets = self.forward(
                actions, states, valid_token_mask, fixed_mask=fixed_mask
            )

            num_steps = (self.args.seq_length - 3) // 2
            action_indices = torch.arange(2, 2 + num_steps * 2, step=2, device=states.device)
            state_indices = torch.arange(3, 3 + num_steps * 2, step=2, device=states.device)
            targets_states = targets[:, state_indices].repeat_interleave(self.repeats, dim=0)
            targets_actions = targets[:, action_indices].repeat_interleave(self.repeats, dim=0)
            valid_aug = valid_token_mask.repeat_interleave(self.repeats, dim=0)
            state_mask = mask[:, state_indices].bool() & valid_aug[:, state_indices]
            action_mask = mask[:, action_indices].bool() & valid_aug[:, action_indices]

            state_losses = F.cross_entropy(
                logits_states.reshape(-1, self.args.state_dim), targets_states.reshape(-1), reduction="none"
            ).reshape_as(state_mask)
            action_losses = F.cross_entropy(
                logits_actions.reshape(-1, self.args.action_dim), targets_actions.reshape(-1), reduction="none"
            ).reshape_as(action_mask)
            state_count = state_mask.sum()
            action_count = action_mask.sum()
            total_count = state_count + action_count
            total_loss = (
                (state_losses * state_mask).sum() + (action_losses * action_mask).sum()
            ) / total_count.clamp(min=1)

            return {
                "loss": total_loss.item(),
                "state_loss": ((state_losses * state_mask).sum() / state_count.clamp(min=1)).item(),
                "action_loss": ((action_losses * action_mask).sum() / action_count.clamp(min=1)).item(),
                "state_tokens": state_count.item(),
                "action_tokens": action_count.item(),
            }
        finally:
            self.train(was_training)
            

    # TODO - test it if it works and make it move forward
    def process_and_augment_batch(self, x, valid_token_mask, fixed_mask=None):
        """x shape: (bsz, seq_len, embed_dim) -> e.g., (4, 11, d_model)

        Returns augmented batch of size (bsz * repeats, seq_len, embed_dim)
        """
        bsz, seq_len, embed_dim = x.shape
        device = x.device

        # 1. Expand batch 4 -> 4 * 64 = 256 samples
        x_aug = x.repeat_interleave(self.repeats, dim=0)  # Shape: (256, 11, D)
        aug_bsz = x_aug.shape[0]

        # 2. Eligible slice: indices 2 to 9 inclusive (8 slots)
        start_idx, end_idx = 2, 10
        num_eligible = end_idx - start_idx  # 8

        valid_aug = valid_token_mask.repeat_interleave(self.repeats, dim=0).bool()
        eligible_valid = valid_aug[:, start_idx:end_idx]

        if fixed_mask is not None:
            if fixed_mask.ndim == 1:
                fixed_mask = fixed_mask.unsqueeze(0).expand(bsz, -1)
            if fixed_mask.shape != (bsz, seq_len):
                raise ValueError(f"fixed_mask must have shape ({seq_len},) or ({bsz}, {seq_len})")
            fixed_mask = fixed_mask.to(device=device, dtype=torch.bool)
            mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
            mask[:, start_idx:end_idx] = (
                fixed_mask[:, start_idx:end_idx] & valid_token_mask[:, start_idx:end_idx].bool()
            )
            mask = mask.repeat_interleave(self.repeats, dim=0)
            p_mask = mask.float().mean(dim=1, keepdim=True).expand(-1, seq_len)
        else:
            eps = 1e-3
            t = torch.rand((aug_bsz, 1), device=device)
            p_mask_scalar = (1 - eps) * t + eps

            if self.mask_strategy == "cosine":
                mask_prob = torch.cos(math.pi / 2 * (1 - p_mask_scalar))
            elif self.mask_strategy == "linear":
                mask_prob = p_mask_scalar
            elif self.mask_strategy == "exp":
                mask_prob = 1 - torch.exp(-5 * p_mask_scalar)
            else:
                raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")

            k_masked = torch.round(1 + mask_prob * (num_eligible - 2)).long().clamp(min=1, max=7)
            valid_count = eligible_valid.sum(dim=1, keepdim=True)
            k_masked = torch.minimum(k_masked, valid_count)

            rand_vals = torch.rand((aug_bsz, num_eligible), device=device)
            rand_vals = rand_vals.masked_fill(~eligible_valid, float("inf"))
            sorted_indices = torch.argsort(rand_vals, dim=-1)
            ranks = torch.argsort(sorted_indices, dim=-1)
            eligible_mask = (ranks < k_masked) & eligible_valid

            mask = torch.zeros((aug_bsz, seq_len), dtype=torch.bool, device=device)
            mask[:, start_idx:end_idx] = eligible_mask
            p_mask = p_mask_scalar.expand(aug_bsz, seq_len)

        return x_aug, mask, p_mask



    def random_masking(self, x):
        bsz, seq_len, embed_dim = x.shape
        
        eps = 1e-3
        valid_mask_generated = False  # Flag to indicate whether a valid mask has been generated

        while not valid_mask_generated:
            t = torch.rand((), device=x.device)
            t = torch.clamp(t, min=self.clamp_t_min)
            p_mask = (1 - eps) * t + eps  # Shared p_mask scalar
            p_mask = p_mask.expand(seq_len)  # Shape is (seq_len,)

            # Generate random_vals based on sequence length
            random_vals = torch.rand((seq_len,), device=x.device)  # Shape is (seq_len,)

            # Select masking strategy
            if self.mask_strategy == 'cosine':  # 'cosine' or 'linear' or 'poly-0.5'
                mask_prob = torch.cos(math.pi / 2 * (1 - p_mask))
            elif self.mask_strategy == 'linear': 
                mask_prob = p_mask
            elif self.mask_strategy == 'poly-0.5':
                mask_prob = p_mask ** 0.5
            elif self.mask_strategy == 'exp':
                mask_prob = 1 - torch.exp(-5 * p_mask)
            elif self.mask_strategy == 'log':
                mask_prob = torch.log(1 + (math.exp(5) - 1) * p_mask) / 5

            # Generate mask
            mask = (random_vals < mask_prob).float()  # Shape is (seq_len,)

            # Check if the number of masks meets the requirement
            if mask.sum() >= 2:
                valid_mask_generated = True

        # Expand the mask to match the batch_size
        mask = mask.expand(bsz, -1)  # Shape is (batch_size, seq_len)
        p_mask = p_mask.expand(bsz, -1)  # Shape is (batch_size, seq_len) 

        random_mask = mask.clone()
        # Shuffle the seq_len order of each sample
        for i in range(bsz):
            random_indices = torch.randperm(seq_len)
            random_mask[i] = random_mask[i][random_indices]
        mask = random_mask

        return mask, p_mask, self.mask_strategy

    # PROBABLY THE ONE TO USE - in my example
    def random_masking_batch(self, x):
        bsz, seq_len, embed_dim = x.shape
        t = torch.rand((bsz,), device=x.device)
        eps = 1e-3
        # we use below to shift the range from 0-1 to 0.001-1 (avoid zero)
        p_mask = (1 - eps) * t + eps  # Shape is (batch_size,)clock
        p_mask = p_mask[:, None].expand(-1, seq_len)  # Shape is (batch_size, seq_len)
        random_vals = torch.rand((bsz, seq_len), device=x.device)
        # Select masking strategy
        if self.mask_strategy == 'cosine':  # 'cosine' or 'linear' or 'poly-0.5'
            mask_prob = torch.cos(math.pi / 2 * (1 - p_mask))
        elif self.mask_strategy == 'linear': 
            mask_prob = p_mask
        elif self.mask_strategy == 'poly-0.5':
            mask_prob = p_mask ** 0.5
        elif self.mask_strategy == 'exp':
            mask_prob = 1 - torch.exp(-5 * p_mask)
        elif self.mask_strategy == 'log':
            mask_prob = torch.log(1 + (math.exp(5) - 1) * p_mask) / 5
        mask = (random_vals < mask_prob).float()
        
        return mask, p_mask, self.mask_strategy


    def random_masking_batch_improved(self, x):
        bsz, seq_len, _ = x.shape
        device = x.device

        # Eligible index range: 2 to 9 inclusive (8 candidate tokens)
        start_idx, end_idx = 2, 10
        num_eligible = end_idx - start_idx  # 8

        # 1. Sample number of tokens to mask per batch element (between 1 and 7)
        num_masked = torch.randint(1, 8, (bsz,), device=device)  # Shape: (bsz,)

        # 2. Generate random values over the eligible slice to determine which specific indices get masked
        rand_vals = torch.rand((bsz, num_eligible), device=device)

        # Get top-k smallest random values per row to pick exact positions randomly
        # topk on -rand_vals or sorting args gives random permutations per row
        _, sorted_indices = torch.sort(rand_vals, dim=-1)

        # Mask positions where rank < num_masked for each row
        ranks = torch.argsort(sorted_indices, dim=-1)
        eligible_mask = (ranks < num_masked.unsqueeze(-1)).float()

        # 3. Construct full sequence mask (11 tokens)
        mask = torch.zeros((bsz, seq_len), device=device)
        mask[:, start_idx:end_idx] = eligible_mask

        # 4. Compute p_mask (fraction of masked tokens over total sequence or eligible slice)
        p_mask = mask.sum(dim=-1, keepdim=True) / seq_len
        p_mask = p_mask.expand(-1, seq_len)

        return mask, p_mask, self.mask_strategy






    ### How to correctly avoid in crossentropy predicting unmasked tokens
    # # Ignore unmasked tokens automatically in loss calculation
    # criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    # # Create targets where unmasked indices are set to -100
    # targets = true_token_ids.clone()
    # targets[mask == 0] = -100  # mask: 1 for masked, 0 for unmasked

    # # Compute loss (automatically divides by total count of masked tokens in the batch)
    # loss = criterion(logits.view(-1, vocab_size), targets.view(-1))





        

