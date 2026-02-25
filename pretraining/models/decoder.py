import torch
import torch.nn as nn 
from pretraining.models.encoder import MVMBlock

class MVMAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, decoder_embed_dim=192, depth=4, num_heads=3, codebook_size = 1024):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias = True)
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1,1568,decoder_embed_dim))

        self.blocks = nn.ModuleList([
            MVMBlock(decoder_embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, codebook_size, bias = True)

    def forward(self, x, mask, ids_restore):
        x = self.decoder_embed(x)
        
        # fill in the blanks with [MASK] token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
        x_all = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_all, dim=1, index = ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))
        
        x = x + self.decoder_pos_embed    
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.pred(x)
        
        return x