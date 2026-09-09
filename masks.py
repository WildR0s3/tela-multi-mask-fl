import itertools
import torch, math
from collections import defaultdict


class Masks:

    def __init__(self):

        self.masks_dict = defaultdict(dict)



    def create_masks(self, sequence_len):

        for i in range(1, 8):
            index_lists = list(itertools.combinations(range(8), i))
            for count, idx_el in enumerate(index_lists, 1):
                mask = torch.ones(sequence_len-3, dtype=torch.long)
                for idx in idx_el:
                    mask[idx] = 0
                self.masks_dict[f"mask_group_{i}"][f"mask_{count}"] = mask




class EpochCombinatorialMasker:

    def __init__(self, repeats=64, mask_strategy="cosine"):
        self.repeats = repeats
        self.mask_strategy = mask_strategy

    def mask_batch(self, x):
        """Called inside your training loop:

        x shape: (4, 11, embed_dim) -> outputs expanded batch (256, 11,
        embed_dim)
        """
        bsz, seq_len, embed_dim = x.shape
        device = x.device

        # Expand sequence 4 -> 256
        x_aug = x.repeat_interleave(self.repeats, dim=0)
        aug_bsz = x_aug.shape[0]

        start_idx, end_idx = 2, 10
        num_eligible = end_idx - start_idx  # 8 slots

        # 1. Schedule-based p_mask -> targets 1..7 tokens
        eps = 1e-3
        t = torch.rand((aug_bsz, 1), device=device)
        p_mask_scalar = (1 - eps) * t + eps

        if self.mask_strategy == "cosine":
            mask_prob = torch.cos(math.pi / 2 * (1 - p_mask_scalar))
        elif self.mask_strategy == "exp":
            mask_prob = 1 - torch.exp(-5 * p_mask_scalar)
        else:
            mask_prob = p_mask_scalar

        # Target mask sizes: 1 to 7 tokens
        k_masked = torch.round(1 + mask_prob * (num_eligible - 2)).long()
        k_masked = torch.clamp(k_masked, min=1, max=7)

        # 2. Draw fresh random permutation per sample (changes every epoch call)
        # rand_vals creates unique rankings for slots [2..9]
        rand_vals = torch.rand((aug_bsz, num_eligible), device=device)
        sorted_indices = torch.argsort(rand_vals, dim=-1)

        # 3. Build binary mask using top-k indices per sample
        ranks = torch.argsort(sorted_indices, dim=-1)
        eligible_mask = (ranks < k_masked).float()

        mask = torch.zeros((aug_bsz, seq_len), device=device)
        mask[:, start_idx:end_idx] = eligible_mask

        p_mask = p_mask_scalar.expand(aug_bsz, seq_len)

        return x_aug, mask, p_mask
