import torch
import torch.nn as nn
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
        
        # Handle whether you saved a raw tensor or a state_dict
        if isinstance(codebook_data, dict):
            # If it's a state_dict, find the actual weight tensor
            # (Adjust 'embedding.weight' to whatever your dict key is)
            key = list(codebook_data.keys())[0] 
            self.codebook = codebook_data[key]
        else:
            # If you saved the raw tensor directly
            self.codebook = codebook_data 
            
        self.codebook = self.codebook.to(device)
        self.codebook.requires_grad = False # STRICTLY FROZEN. Do not update during pre-training.

        # Store the dimension (e.g., 256 or 384)
        self.embed_dim = self.codebook.shape[1] 

        # Optional: If you used a VQ-VAE Encoder to compress the MVs *before* # hitting the codebook, you would load that frozen encoder here.
        # self.vq_encoder = MyFrozenVQEncoder().to(device)
        # self.vq_encoder.eval()

    @torch.no_grad() # Crucial: No gradients should flow into the tokenizer
    def tokenize(self, mvs):
        """
        Calculates the Euclidean distance between your input MVs and the 1024 Codebook vectors,
        returning the ID of the closest match.
        """
        # Step 1: Feature Extraction
        # If your codebook operates directly on the flattened 2-channel patches:
        # (Assuming your codebook was trained on vectors of shape [2 * 16 * 14 * 14] or similar)
        # You need to reshape `mvs` to match the exact dimension your codebook expects.
        
        # Example: Let's assume you have a frozen VQ-encoder that turns MVs into latents
        # latents = self.vq_encoder(mvs) # -> [B, 1568, embed_dim]
        
        # For this example, let's assume `latents` is already the right shape [B, 1568, embed_dim]
        # (You will need to adjust this one line to match how your codebook was trained)
        latents = mvs.view(mvs.shape[0], 1568, -1) 
        
        # Step 2: Calculate L2 Distance mathematically
        # distance = (x - y)^2 = x^2 - 2xy + y^2
        latents_squared = (latents ** 2).sum(dim=-1, keepdim=True)
        codebook_squared = (self.codebook ** 2).sum(dim=-1)
        
        # Matrix multiplication for the 2xy term
        cross_term = torch.matmul(latents, self.codebook.t())
        
        # Final distance [B, 1568, 1024]
        distances = latents_squared - 2 * cross_term + codebook_squared
        
        # Step 3: Find the closest Codebook ID (argmin)
        # This returns the index (0 to 1023) of the nearest motion concept
        target_ids = torch.argmin(distances, dim=-1) # Shape: [B, 1568]
        
        return target_ids