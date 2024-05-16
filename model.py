import torch
import torch.nn as nn
from torch.nn import functional as F

d_model = 32 # dimension of the intermediate embedding vectors
block_size = 64 # context length

class Head(nn.Module):
    def __init__(self, h_size):
        super().__init__()
        self.key = nn.Linear(d_model, h_size, bias=False)
        self.query = nn.Linear(d_model, h_size, bias=False)
        self.value = nn.Linear(d_model, h_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        b, t, e = x.size()
        k = self.key(x)
        q = self.query(x)

        # compute the attention weights
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        
        # weighted aggregation of values
        v = self.value(x)
        return w @ v


class MultiHead(nn.Module):
    def __init__(self, n_heads, h_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(h_size) for _ in range(n_heads)])
        self.proj = nn.Linear(h_size * n_heads, d_model)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(self, x):
        return self.net(x)

class GPT():
    def __init__(self):
        super().__init__()
