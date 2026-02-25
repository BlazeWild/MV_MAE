#encoder.py
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class MVMBlock(nn.Module):
    """Optimized transformer block for motion vectors"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first = True
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MVMAEEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        # Input : [B,2,16,16,14,14] -> Output: [B, 384, 8, 14, 14]
        # we use 1x1 spatial kernels as data is already 14x14
        self.patch_embed = nn.Conv3d(
            in_channels=2,
            out_channels=embed_dim,
            kernel_size=(2,1,1),
            stride=(2,1,1)
        )
        # 8 steps * 14 height = 1566 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 1568, embed_dim))
        self.blocks = nn.ModuleList([
            MVMBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def random_masking(self, x, mask_ratio):
        B,N,L = x.shape
        len_keep = int(N*(1-mask_ratio))

        noise = torch.rand(B,N,device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,L))
        
        # Force the mask to inherit the exact dtype (bfloat16/float32) of the input tensor
        mask = torch.ones([B,N], device=x.device, dtype=x.dtype, requires_grad=False)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.9):
        # flatten batch and sequence if 6D
        if x.dim() == 6:
            B,S,C,T,H,W = x.shape
            x = x.view(B*S, C, T, H, W)
        
        x = self.patch_embed(x) # [B*S, 384, 8, 14, 14]
        x = x.flatten(2).transpose(1,2) # [B*S, 1568, 384]
        x = x + self.pos_embed
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        return x, mask, ids_restore