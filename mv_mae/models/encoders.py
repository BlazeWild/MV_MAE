import torch 
import torch.nn as nn
import timm
from transformers import VideoMAEModel, VideoMAEConfig
import logging
import os

logger = logging.getLogger(__name__)

class ContextEncoder(nn.Module):
    """
    Encodes the I-frame using a lightweight ViT-Tiny.
    Input -> [B, 3, 224, 224]
    Output -> [B, 192] (the class token features)
    """
    def __init__(self, model_zoo_path: str = "./model_zoo"):
        super().__init__()
        
        weights_path = os.path.join(model_zoo_path, "clip_context/vit_tiny_patch16_224.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Context Encoder Weights not found at {weights_path}") 

        logger.info("Loading ViT-Tiny Context Encoder")
        self.vit = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False, num_classes=0)

        # Load weights manually 
        state_dict = torch.load(weights_path, map_location="cpu")
        # Remove classification head weights if present, as we set num_classes=0
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        self.vit.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        return self.vit(x)

class MotionEncoder(nn.Module):
    """
    Encodes the Motion Vectors using VideoMAE-Small.
    Input -> [B, 2, 16, H, W]
    Output -> [B, 384] (the mean pool features)
    """
    def __init__(self, model_zoo_path: str = "./model_zoo"):
        super().__init__()
        
        model_dir = os.path.join(model_zoo_path, "video_mae")
        weights_path = os.path.join(model_dir, "vit_s_k400_ft.pth")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"VideoMAE weights not found at {weights_path}")

        logger.info("Loading VideoMAE Motion Encoder")
        
        # FIX: Load config from the directory containing config.json, not the .pth file
        config = VideoMAEConfig.from_pretrained(model_dir)
        self.videomae = VideoMAEModel(config)

        # Load weights manually
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.videomae.load_state_dict(state_dict, strict=False)

    def forward(self, mvs):
        # [B, 2, 16, H, W] -> [B, 16, 2, H, W] for VideoMAE
        x = mvs.permute(0, 2, 1, 3, 4) 
        outputs = self.videomae(x)
        
        # Last hidden state: [Batch, Sequence_Length, Hidden_Dim]
        last_hidden_state = outputs.last_hidden_state 

        # Mean pooling across the sequence dimension to get a single vector per clip
        mot_feat = torch.mean(last_hidden_state, dim=1)
        return mot_feat