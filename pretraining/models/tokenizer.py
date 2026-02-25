#tokenizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MVCodebookTokenizer(nn.Module):
    """
    Loads your pre-trained 1024-token codebook and maps continuous 
    motion vectors to discrete IDs using L2 Distance.
    """
    def __init__(self, codebook_path='../codebook_ckpt/mv_codebook_1024.pt', device='cuda'):
        super().__init__()
        
        # 1. Load the Codebook
        logger.info(f"Loading real MV codebook from: {codebook_path}")
        codebook_data = torch.load(codebook_path, map_location=device)
        
        if isinstance(codebook_data, dict):
            key = list(codebook_data.keys())[0] 
            self.codebook = codebook_data[key]
        else:
            self.codebook = codebook_data 
            
        self.codebook = self.codebook.to(device).float()
        self.codebook.requires_grad = False # STRICTLY FROZEN.

        self.embed_dim = self.codebook.shape[1] 

    @torch.no_grad()
    def tokenize(self, mvs):
        """
        Calculates the Euclidean distance between your input MVs and the 1024 Codebook vectors,
        returning the ID of the closest match.
        """
        # Step 1: Handle 6D Dataloader Batches
        # If input is [Batch, Segments, Channels, Time, Height, Width]
        if mvs.dim() == 6:
            B, S, C, T, H, W = mvs.shape
            mvs = mvs.view(B * S, C, T, H, W)
            
        # Step 2: Temporal Pooling (THE CRITICAL FIX)
        # We must compress 16 frames to 8 steps so it perfectly matches the 1568 sequence length of the Encoder
        # [BS, 2, 16, 14, 14] -> [BS, 2, 8, 14, 14]
        mvs_pooled = F.avg_pool3d(mvs, kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # Step 3: Permute and Flatten
        BS, C_p, T_p, H_p, W_p = mvs_pooled.shape
        # Move channel dimension to the end: [BS, 8, 14, 14, 2]
        latents = mvs_pooled.permute(0, 2, 3, 4, 1).contiguous()
        # Flatten into exactly 1568 tokens: [BS, 1568, 2]
        latents = latents.view(BS, T_p * H_p * W_p, C_p)
        
        # Step 4: Calculate L2 Distance mathematically
        latents_squared = (latents ** 2).sum(dim=-1, keepdim=True)
        codebook_squared = (self.codebook ** 2).sum(dim=-1)
        
        # cross_term: [BS, 1568, 1024]
        cross_term = torch.matmul(latents, self.codebook.t())
        
        # Final distance [BS, 1568, 1024]
        distances = latents_squared - 2 * cross_term + codebook_squared
        
        # Step 5: Find the closest Codebook ID (argmin)
        target_ids = torch.argmin(distances, dim=-1) # Shape: [BS, 1568]
        
        return target_ids