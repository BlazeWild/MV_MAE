import torch
import torch.nn as nn
from collections import OrderedDict

class QuickGELU(nn.Module):
    """A faster approximation of GELU used in standard CLIP/Transformer models."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """A standard Transformer Encoder Block (Self-Attention + MLP) using Pre-LayerNorm."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        
        # Define LayerNorms
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
        # Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True)
        
        # Feed-Forward MLP
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout_1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout_2", nn.Dropout(dropout))
        ]))

    def forward(self, x: torch.Tensor):
        # x shape: [Batch, Seq_Len, d_model]
        
        # 1. Self-Attention Block (Pre-LN)
        normalized_x1 = self.ln_1(x)
        attn_out, _ = self.attn(normalized_x1, normalized_x1, normalized_x1)
        x = x + attn_out
        
        # 2. Feed-Forward Block (Pre-LN)
        normalized_x2 = self.ln_2(x)
        mlp_out = self.mlp(normalized_x2)
        x = x + mlp_out
        
        return x

class TemporalTransformer(nn.Module):
    """
    Acts as the 'Decoder', processing the sequence of N fused GOP features over time.
    """
    def __init__(self, d_model: int, n_layers: int, n_head: int, max_gops: int = 16, dropout: float = 0.1):
        super().__init__()
        
        # A learned token representing the entire video clip
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional embedding so the model knows the temporal order of the GOPs
        self.positional_embedding = nn.Parameter(torch.randn(1, max_gops + 1, d_model) / (d_model ** 0.5))
        
        # The stack of Transformer blocks
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(d_model, n_head, dropout) for _ in range(n_layers)
        ])
        
        self.ln_post = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [Batch, Num_GOPs, d_model]
        """
        batch_size, num_gops, _ = x.shape
        
        # 1. Add the Temporal CLS Token to the beginning of the sequence
        cls_tokens = self.temporal_cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: [Batch, Num_GOPs + 1, d_model]
        
        # 2. Add Positional Embeddings
        x = x + self.positional_embedding[:, :num_gops + 1, :]
        
        # 3. Pass through all Transformer layers
        for block in self.resblocks:
            x = block(x)
            
        # 4. Final normalization
        x = self.ln_post(x)
        
        # 5. Extract ONLY the CLS token feature (this now contains the temporal context of the whole video)
        video_level_feature = x[:, 0, :] # Shape: [Batch, d_model]
        
        return video_level_feature

