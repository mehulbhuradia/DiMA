import torch
import torch.nn.functional as F
from torch import nn

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class HalfTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim=config.hidden_size
        context_dim=config.cross_context_dim
        n_heads=config.num_attention_heads
        gated_ff=config.cross_gated_ff
        
        self.ff = FeedForward(dim, glu=gated_ff)
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):
        x = self.attn(self.norm1(x), key=context, value=context, need_weights=False)[0] + x
        x = self.ff(self.norm2(x)) + x
        return x
