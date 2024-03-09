import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout=0.3, max_seq_len=256, device="cpu"):
        super().__init__()
        self.d = d  # dimensions of the model

        self.dropout = nn.Dropout(dropout)

        self.pe = torch.zeros(max_seq_len, self.d, device=device)
        self.pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, max_seq_len, step=2).float()
        div = torch.pow(10000, two_i / torch.tensor([self.d])).float()
        self.pe[:, 0::2] = torch.sin(self.pos / div)  # even indices 
        self.pe[:, 1::2] = torch.cos(self.pos / div)  # odd indices
        
    def forward(self, x):
        # x.shape = [batch_size x seq_len x d]
        first_batch_pe = self.pe[:, :x.shape[1]].detach()
        repeated_pe = first_batch_pe.repeat([x.shape[0], 1, 1]).detach()
        
        x = x.add(repeated_pe)
        
        return self.dropout(x)
        

