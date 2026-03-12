import torch 
import torch.nn as nn
import timm
import logging
import os

logger = logging.getLogger(__name__)

class ContextEncoder(nn.Module):
    """
    Encodes the I-frame using a ViT-Base.
    Input -> [B, 3, 224, 224]
    Output -> [B, 768] (the class token features)
    """
    def __init__(self, model_zoo_path: str = "./model_zoo"):
        super().__init__()
        
        weights_path = os.path.join(model_zoo_path, "clip_context/vit_base_patch16_224.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Context Encoder Weights not found at {weights_path}") 

        logger.info("Loading ViT-Base Context Encoder")
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

        # Load weights manually 
        state_dict = torch.load(weights_path, map_location="cpu")
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Remove classification head weights if present, as we set num_classes=0
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        self.vit.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        return self.vit(x)

class MotionEncoder(nn.Module):
    """
    Encodes Motion Vectors using the pretrained MVMAEEncoder from DMVMAE pretraining.
    Loads only the encoder weights from the full pretraining checkpoint.
    
    Input  -> [B, 2, 16, 14, 14]  (raw MV grids, NO upsampling, NO tokenization)
    Output -> [B, 384]            (mean-pooled encoder features)
    """
    def __init__(self, pretrain_ckpt_path: str):
        super().__init__()
        
        from pretraining.models.encoder import MVMAEEncoder
        
        self.encoder = MVMAEEncoder(embed_dim=384, depth=12, num_heads=6)
        
        # Load only encoder weights from full checkpoint
        if not os.path.exists(pretrain_ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found at {pretrain_ckpt_path}")
        
        logger.info(f"Loading pretrained MVMAEEncoder from: {pretrain_ckpt_path}")
        checkpoint = torch.load(pretrain_ckpt_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            full_state = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', '?')
            logger.info(f"  Checkpoint epoch: {epoch}")
        else:
            full_state = checkpoint
        
        # Extract only encoder.* keys and strip the "encoder." prefix
        encoder_state = {k.replace("encoder.", "", 1): v 
                         for k, v in full_state.items() if k.startswith("encoder.")}
        
        msg = self.encoder.load_state_dict(encoder_state, strict=True)
        logger.info(f"  Encoder weights loaded: {msg}")

    def forward(self, mvs):
        # mvs: [B, 2, 16, 14, 14] — raw MV grids
        # mask_ratio=0.0 means NO masking during finetuning (use all tokens)
        latent, _, _ = self.encoder(mvs, mask_ratio=0.0)
        
        # latent: [B, 1568, 384] — mean pool across token dimension
        mot_feat = torch.mean(latent, dim=1)  # [B, 384]
        return mot_feat