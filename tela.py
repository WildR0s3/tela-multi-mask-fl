
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_modules.encoder import EncoderBlock
from logger.logging_module import log, lte
from masks import Masks
import copy
from collections import defaultdict

EMBEDING_DIM = 8
ACTION_DIM = 4
STATE_DIM = 16 + 3 # states + TERM + START, PAD will be masked sbut let's add it here to allow full abtch embedding alogn with tid
DROP_OUT = 0.0
NUM_OF_HEADS = 8
NUM_OF_LAYERS = 2
LEARNING_RATE = 1e-4
SEQ_LENGTH = 11
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
torch.set_printoptions(precision=4, sci_mode=False)

class TELA(nn.Module):

    def __init__(self):
        super().__init__()
        self.states_embed = nn.Embedding(STATE_DIM, EMBEDING_DIM)
        self.action_embed = nn.Embedding(ACTION_DIM, EMBEDING_DIM)

        self.mask_action_token = nn.Parameter(torch.zeros(1, 1, EMBEDING_DIM))
        nn.init.normal_(self.mask_action_token, std=0.02)
        self.mask_state_token = nn.Parameter(torch.zeros(1, 1, EMBEDING_DIM))
        nn.init.normal_(self.mask_state_token, std=0.02)

        self.pos_embedding = nn.Embedding(SEQ_LENGTH, EMBEDING_DIM)

        self.blocks = nn.ModuleList([EncoderBlock(EMBEDING_DIM, NUM_OF_HEADS, DROP_OUT) for _ in range(NUM_OF_LAYERS)])

        self.ln_f = nn.LayerNorm(EMBEDING_DIM)
        self.action_head = nn.Linear(EMBEDING_DIM, ACTION_DIM)
        self.states_head = nn.Linear(EMBEDING_DIM, STATE_DIM)

        self.criterion = nn.CrossEntropyLoss()
        self.model_optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE) # weight_decay=1e-4 // for a small model remove

        m = Masks()
        m.create_masks(SEQ_LENGTH)
        self.masks = m.masks_dict



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




        

