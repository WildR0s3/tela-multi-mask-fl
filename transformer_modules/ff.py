import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embeding_dim, drop_out):
        super().__init__()
        # increase the dimension to 4 times and then reduce it back to increase computational capactiy of feed forward
        self.net = nn.Sequential(nn.Linear(embeding_dim, 4 *embeding_dim),
                                 nn.GELU(),
                                 nn.Linear(4 * embeding_dim, embeding_dim),
                                 nn.Dropout(drop_out),)
        
    def forward(self, x):
        return self.net(x)