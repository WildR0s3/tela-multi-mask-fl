import itertools
import torch
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
