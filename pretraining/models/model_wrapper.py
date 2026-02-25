#model_wrapper.py
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pretraining.models.encoder import MVMAEEncoder
from pretraining.models.decoder import MVMAEDecoder

class DiscreteMVMAE(nn.Module):
    def __init__(self, codebook_size = 1024):
        super().__init__()
        self.encoder = MVMAEEncoder()
        self.decoder = MVMAEDecoder(codebook_size=codebook_size)

    def forward(self, x, target_ids, mask_ratio=0.9):
        """
        mvs : [B,8,2,16,14,14]
        target_ids: [B*8, 1568] <- Pre-calcuated from VQ-Tokeinizer
        """

        # 1. Encode only the 10percent visible tokens
        latent, mask, ids_restore = self.encoder(x, mask_ratio)
        
        # decode/predict the masked_tokens
        logits = self.decoder(latent,ids_restore)
        
        # 2. Calculate CrossEntropy loss
        # we only calculate loss on the masked tokens (mask==1)
        if target_ids.dim() == 3: # handle batch if needed
            target_ids = target_ids.view(-1, 1568)
        
        loss = F.cross_entropy(logits.view(-1, 1024), target_ids.view(-1), reduction = 'none')

        # only apply_loss where mask==1 
        loss = (loss.view(logits.shape[0], -1) * mask).sum() / mask.sum()
        return loss, logits