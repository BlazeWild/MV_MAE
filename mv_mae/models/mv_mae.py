import torch
import torch.nn as nn
from .encoders import ContextEncoder, MotionEncoder
from .fusion import MultiModalFusion

class MvMaeNET(nn.Module):
    """
    GreenEyeNet: Efficient Spatio-Temporal Action Recognition
    Input: 
      - I-Frame (Context): [B, 3, 224, 224]
      - Motion Vectors (Action): [B, 2, 16, 224, 224]
    Output:
      - Class Logits: [B, Num_Classes]
    """
    def __init__(self, num_classes=10, drop_path=0.1):
        super().__init__()
        
        # 1. The "Eye" (Context Stream)
        # Uses ViT-Tiny to see the scene (Park, Indoor, Sky)
        self.context_stream = ContextEncoder(model_name='vit_tiny_patch16_224')
        
        # 2. The "Motion" (Action Stream)
        # Uses VideoMAE-Small to see the movement (Running, Waving)
        self.motion_stream = MotionEncoder(model_name='videomae_small', frames=16)
        
        # 3. The "Brain" (Fusion)
        # Fuses the two streams (192 dim + 384 dim)
        self.fusion_module = MultiModalFusion(
            dim_ctx=192, 
            dim_mot=384, 
            hidden_dim=256, 
            num_classes=num_classes
        )

    def forward(self, iframe, mvs):
        """
        Forward pass of the Green-Eye Network.
        """
        # --- A. Encode Context ---
        # iframe: [Batch, 3, 224, 224]
        # ctx_feat: [Batch, 192]
        ctx_feat = self.context_stream(iframe)
        
        # --- B. Encode Motion ---
        # mvs: [Batch, 2, 16, 67, 120] -> We resize inside the encoder
        # mot_feat: [Batch, 384]
        mot_feat = self.motion_stream(mvs)
        
        # --- C. Fuse & Classify ---
        logits = self.fusion_module(ctx_feat, mot_feat)
        
        return logits