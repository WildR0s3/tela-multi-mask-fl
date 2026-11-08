import torch


def print_actions(actions_idx, action_map):
    actions_list = []
    for action in actions_idx[0]:
        actions_list.append(action_map[action.item()])
    print(f"{actions_idx[0]}")
    print(actions_list)


def create_attention_mask(input_tensor, seq_len=11):
    batch_size = input_tensor.shape[0]
    num_padding = (input_tensor == -1).sum(dim=1)
    mask_count = num_padding * 2
    indices = torch.arange(seq_len, device=input_tensor.device)
    mask = torch.ones(seq_len, dtype=torch.float)
    thresholds = seq_len - mask_count.unsqueeze(1)

    mask = (indices < thresholds).float()
    
    return mask, mask_count


### OPTIMZER LEARNING RATE CONTROL
# import torch

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# for batch_idx, data in enumerate(dataloader):
#     # Your training logic here...
    
#     # Manual override: After 500 batches, drop LR
#     if batch_idx == 500:
#         print("Lowering learning rate...")
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-4  # Set new LR


