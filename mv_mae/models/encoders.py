import torch 
import torch.nn as nn
import timm
from transformers import VideoMAEModel, VideoMAEConfig
import logging
import os

logger = logging.getLogger(__name__)

class ContextEncoder(nn.Module):
    """
    Encodes the I-frame using a lightweight Vit-Tiny 
    Input -> B, 3, 224, 224
    Output -> B, 192 (the class token features)
    """
    def __init__(self, model_zoo_path:str="./model_zoo"):
        super().__init__()
        
        weights_path = os.path.join(model_zoo_path, "clip_context/vit_tiny_patch16_224.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Context Encoder Weights not found at {weights_path}") 

        logger.info("Loading Vit-Tiny Context Encoder")
        self.vit = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False, num_classes=0)

        #load weights manually 
        state_dict = torch.load(weights_path, map_location="cpu")
        # remove classification head weights if present, as we set num_classes =0
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith("head.")}
        self.vit.load_state_dict(state_dict)

    def forward(self, x):
        # shape: batch, 3, h, w
        # output: batch, 192
        return self.vit(x)

class MotionEncoder(nn.Module):
    """
    Encodes the Motion Vectors using VideoMAE-Small
    Input ->[ B, 2, 16 , H, W ],  this H,W is eventually 224, 224 resized from original of 67, 120
    Output -> B, 384 (the mean pool features)
    """
    def __init__(self, model_zoo_path:str="./model_zoo"):
        super().__init__()
        
        model_dir = os.path.join(model_zoo_path, "video_mae")
        weights_path = os.path.join(model_dir, "vit_s_k400_ft.pth")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"VidoeMAE weights not found at {weights_path}")

        logger.info("Loading VideoMAE Motion Encoder")
        config = VideoMAEConfig.from_pretrained(weights_path)

        self.videomae = VideoMAEModel(config)

        #load weights manually
        state_dict = torch.load(weights_path, map_location='cpu')
        # VideoMAEModel expects a sepecific prefix for the weights
        if 'model' in state_dict:
            state_dict  =state_dict['model']
        self.videomae.load_state_dict(state_dict, strict=False)

    def forward(Self, mvs):
        """Args:
            mvs: [B, 2, 16, H, W]
        """
        x = mvs.permute(0,2,1,3,4) # [B, 16, 2, H, W] -> [B, 16, 2, H, W]
        outputs = self.videomae(x)
        #last hidden state as (batch, t*h*w/p^2, hidden_dim)
        last_hidden_state = outputs.last_hidden_state 

        #mean pooling to get a single vector clip
        # shape: [batch, 384]
        mot_feat = torch.mean(last_hidden_state, dim=1)

        return mot_feat

        