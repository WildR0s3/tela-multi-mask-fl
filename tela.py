
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

EMBEDING_DIM    = 8
ACTION_DIM      = 4
STATE_DIM       = 16 + 3 # states + TERM + START, PAD will be masked sbut let's add it here to allow full batch embedding alogn with tid
DROP_OUT        = 0.0
NUM_OF_HEADS    = 8
NUM_OF_LAYERS   = 2
LEARNING_RATE   = 1e-4
SEQ_LENGTH      = 11
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE      = 4
torch.set_printoptions(precision=4, sci_mode=False)

class TELA(nn.Module):

    def __init__(self):
        super().__init__()
        self.states_embed = nn.Embedding(STATE_DIM, EMBEDING_DIM)
        self.action_embed = nn.Embedding(ACTION_DIM, EMBEDING_DIM)
        self.mask_strategy = 'exp'

        self.mask_action_token = nn.Parameter(torch.zeros(1, 1, EMBEDING_DIM))
        nn.init.normal_(self.mask_action_token, std=0.02)
        self.mask_state_token = nn.Parameter(torch.zeros(1, 1, EMBEDING_DIM))
        nn.init.normal_(self.mask_state_token, std=0.02)

        self.pos_embedding = nn.Embedding(SEQ_LENGTH, EMBEDING_DIM)
        self.action_type_embed = nn.Parameter(torch.randn(1, 1, EMBEDING_DIM))
        nn.init.normal_(self.action_type_embed, std=0.02)
        self.state_type_embed = nn.Parameter(torch.randn(1, 1, EMBEDING_DIM))
        nn.init.normal_(self.state_type_embed, std=0.02)

        self.blocks = nn.ModuleList([EncoderBlock(EMBEDING_DIM, NUM_OF_HEADS, DROP_OUT) for _ in range(NUM_OF_LAYERS)])

        self.ln_f = nn.LayerNorm(EMBEDING_DIM)
        self.action_head = nn.Linear(EMBEDING_DIM, ACTION_DIM)
        self.states_head = nn.Linear(EMBEDING_DIM, STATE_DIM)

        self.criterion = nn.CrossEntropyLoss()
        self.model_optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE) # weight_decay=1e-4 // for a small model remove

        m = Masks()
        m.create_masks(SEQ_LENGTH)
        self.masks = m.masks_dict
        self.repeats = 16



    def forward(self, actions, states, padding_mask):
        states_emb = self.states_embed(states)
        actions_emb = self.action_embed(actions)
        # for i in range(BATCH_SIZE // 8):
        sequence = torch.zeros((BATCH_SIZE, SEQ_LENGTH, EMBEDING_DIM), device=DEVICE)

        targets = torch.zeros((BATCH_SIZE, SEQ_LENGTH), device=DEVICE)
        targets[:, :2] = states[:, :2]
        targets[:, 2:10:2] = actions
        targets[:, 3::2] = states[:, 2:6]
        targets[:, -1] = states[:, -1]

        sequence[:, 0] = states_emb[:, 0, :]
        sequence[:, 1] = states_emb[:, 1, :]

        num_steps = (SEQ_LENGTH - 3) // 2

        action_indices = torch.arange(2, 2 + num_steps * 2, step=2, device=DEVICE)
        state_indices  = torch.arange(3, 3 + num_steps * 2, step=2, device=DEVICE)

        sequence[:, action_indices] = actions_emb[:, :num_steps, :]
        sequence[:, state_indices]  = states_emb[:, :num_steps, :]

        # for j in range((SEQ_LENGTH - 3) // 2):
        #     action_val = actions_emb[0 , j]
        #     sequence[(j+1)*2] = action_val
        #     state_val = states_emb[0, j]
        #     sequence[((j+1)*2)+1] = state_val

        sequence[:, 10] = states_emb[:, 6, :]

        x_aug, mask, p_mask = self.process_and_augment_batch(sequence)

        indices = torch.arange(SEQ_LENGTH, device=DEVICE)
        is_even = ((indices % 2 == 0).unsqueeze(0).unsqueeze(-1))
        to_be_masked = torch.where(is_even, self.mask_action_token, self.mask_state_token)
        # TODO - to be verified, especially the size if troch arrange is needed
        type_embed = torch.where(is_even, self.action_type_embed, self.state_type_embed) 

        mask_expanded = mask.unsqueeze(-1)
        x_masked = torch.where(mask_expanded == 1.0, to_be_masked, x_aug)
        position_embed = self.pos_embedding(torch.arange(SEQ_LENGTH, device=DEVICE))
        # TODO - to be verified addign type embed
        x = x_masked + position_embed + type_embed 
        padding_mask_aug = padding_mask.repeat_interleave(self.repeats, dim=0)
        for block in self.blocks:
            x = block(x, padding_mask_aug)
        x = self.ln_f(x)

        return x_aug, mask, p_mask, targets

    # There are 2 alternatives - below is flattening, or i I can use ignore index in F.cross_entropy
    def learn_model(self, logits, targets, mask, p_mask=None):
        logits_flat = logits.reshape(-1, EMBEDING_DIM)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Apply optional eMIGM schedule weighting: weight = 1 / p_mask
        if p_mask is not None:
            p_mask_flat = p_mask.reshape(-1)
            loss_per_token = loss_per_token / torch.clamp(p_mask_flat, min=1e-3)

        masked_loss = loss_per_token * mask_flat
        total_masked_tokens = mask_flat.sum()
        loss = masked_loss.sum()
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
    

    # TODO - test it if it works and make it move forward
    def process_and_augment_batch(self, x):
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

        # 3. Sample timesteps t and compute p_mask per augmented sample
        eps = 1e-3
        t = torch.rand((aug_bsz, 1), device=device)
        p_mask_scalar = (1 - eps) * t + eps

        if self.mask_strategy == "cosine":
            mask_prob = torch.cos(math.pi / 2 * (1 - p_mask_scalar))
        elif self.mask_strategy == "linear":
            mask_prob = p_mask_scalar
        elif self.mask_strategy == "exp":
            mask_prob = 1 - torch.exp(-5 * p_mask_scalar)

        # Map p_mask to 1..7 tokens out of 8 slots
        k_masked = torch.round(1 + mask_prob * (num_eligible - 2)).long()
        k_masked = torch.clamp(k_masked, min=1, max=7)

        # 4. Generate unique random mask positions for each repeat
        rand_vals = torch.rand((aug_bsz, num_eligible), device=device)
        _, sorted_indices = torch.sort(rand_vals, dim=-1)
        ranks = torch.argsort(sorted_indices, dim=-1)

        eligible_mask = (ranks < k_masked).float()

        # 5. Build final mask
        mask = torch.zeros((aug_bsz, seq_len), device=device)
        mask[:, start_idx:end_idx] = eligible_mask

        p_mask = p_mask_scalar.expand(aug_bsz, seq_len)

        return x_aug, mask, p_mask



    def assemble_masked_sequences(self, states, actions):

        states_emb = self.states_embed(states)
        actions_emb = self.action_embed(actions)
        canvas = torch.zeros((BATCH_SIZE, SEQ_LENGTH, EMBEDING_DIM), device=DEVICE)



    def batch_preparation(self, actions, states):
        states_emb = self.states_embed(states)
        actions_emb = self.action_embed(actions)
        # for i in range(BATCH_SIZE // 8):
        sequence = torch.zeros((SEQ_LENGTH, EMBEDING_DIM), device=DEVICE)

        sequence[0] = states_emb[0, 0].unsqueeze(0)
        sequence[1] = states_emb[0, 1].unsqueeze(0)

        for j in range((SEQ_LENGTH - 3) // 2):
            action_val = actions_emb[0,j]
            sequence[(j+1)*2] = action_val
            state_val = states_emb[0, j]
            sequence[((j+1)*2)+1] = state_val

        sequence[10] = states_emb[0, 6].unsqueeze(0)

        masked_seqs = defaultdict(dict)
        indices = torch.arange(SEQ_LENGTH - 3)

        for group_key, mask_groups in self.masks.items():
            log(f"Processing {group_key}")
            tmp_list = []
            for mask_key, mask_tensor in mask_groups.items():
                sequence_clone = sequence.clone() # to be verified if this is correct
                sub_sequence = sequence_clone[2:10]
                # if group_key == 'mask_group_3':
                #     print()
                ## below possibly to be improved with .where
                sub_sequence[(mask_tensor == 0) & (indices % 2 == 0)] = self.mask_action_token
                sub_sequence[(mask_tensor == 0) & (indices % 2 != 0)] = self.mask_state_token
                tmp_list.append(sequence_clone)
            masked_seqs[group_key] = tmp_list
              
        all_batches = []
        for batch_idx in range(8):
            batch_elements = []
            for group_key in self.masks.keys():
                group_tensors = masked_seqs[group_key]
                num_available = len(group_tensors)
                for i in range(8):
                    # we wrap around to get back to the beginning
                    idx = (batch_idx * 8 + i) % num_available 
                    batch_elements.append(group_tensors[idx])
            batch_elements.extend(masked_seqs['mask_group_4'][64:])
            batch_elements.extend(masked_seqs['mask_group_4'][68:])

            current_batch = torch.stack(batch_elements, dim=0)
            all_batches.append(current_batch)

        return all_batches

        # cloned_masked_seqs = {key: [t.clone() for t in tensor_list] for key, tensor_list in masked_seqs.items()}
        # # TODO - convert batch list into a tensor with batch list as first batch element
        # # TODO - add further batches up to 8, with popping from the batches and then again reiterating over those the same as in excell
        # batch_list = []
        # batch_list.extend(masked_seqs['mask_group_1'])
        # for i in range(8):
        #     batch_list.append(cloned_masked_seqs['mask_group_2'].pop(i))
        #     batch_list.append(cloned_masked_seqs['mask_group_3'].pop(i))
        #     batch_list.append(cloned_masked_seqs['mask_group_4'].pop(i))
        #     batch_list.append(cloned_masked_seqs['mask_group_5'].pop(i))
        #     batch_list.append(cloned_masked_seqs['mask_group_6'].pop(i))
        # batch_list.extend(masked_seqs['mask_group_7'])
        # batch_list.extend(masked_seqs['mask_group_4'][64:]) # add always last 6 that will be skipped
        # batch_list.extend(masked_seqs['mask_group_4'][68:]) # and then again last to to have nice 8

        print(f"")


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







    def random_masking(self, x):
        bsz, seq_len, embed_dim = x.shape
        eps = 1e-3
        
        # We will create a mask for each sample individually
        # to ensure every example in the batch is different
        final_masks = []
        p_mask_values = []

        for _ in range(bsz):
            valid_mask_generated = False
            while not valid_mask_generated:
                t = torch.rand((), device=x.device)
                t = torch.clamp(t, min=self.clamp_t_min)
                p_val = (1 - eps) * t + eps 

                # Strategy selection (same as your original code)
                if self.mask_strategy == 'cosine':
                    mask_prob = torch.cos(math.pi / 2 * (1 - p_val))
                elif self.mask_strategy == 'linear': 
                    mask_prob = p_val
                # ... (include other elifs here) ...

                random_vals = torch.rand((seq_len,), device=x.device)
                mask = (random_vals < mask_prob).float()

                if mask.sum() >= 2:
                    valid_mask_generated = True
                    final_masks.append(mask)
                    p_mask_values.append(p_val)

        # Stack everything back into tensors
        mask = torch.stack(final_masks) # Shape (bsz, seq_len)
        p_mask = torch.stack(p_mask_values).unsqueeze(-1).expand(bsz, seq_len) 

        return mask, p_mask, self.mask_strategy



    ### How to correctly avoid in crossentropy predicting unmasked tokens
    # # Ignore unmasked tokens automatically in loss calculation
    # criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

    # # Create targets where unmasked indices are set to -100
    # targets = true_token_ids.clone()
    # targets[mask == 0] = -100  # mask: 1 for masked, 0 for unmasked

    # # Compute loss (automatically divides by total count of masked tokens in the batch)
    # loss = criterion(logits.view(-1, vocab_size), targets.view(-1))





        

