import torch 
import torch.nn as nn
from .encoders import ContextEncoder, MotionEncoder
from .temporal import TemporalTransformer

class ClassificationHead(nn.Module):
    """Takes the final video-level feature and outputs class predictions."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 155, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class MVMAE(nn.Module):
    """
    MV-MAE Complete Hierarchical Architectire"""
    def __init__(self, num_classes=155, model_zoo_path="./model_zoo",num_temporal_layers=2, n_head=8, dim_ctx=192, dim_mot=384, max_gops=16):
        super().__init__()
        # spatial and motion backbones
        self.ctx_encoder = ContextEncoder(model_zoo_path)
        self.mot_encoder = MotionEncoder(model_zoo_path)
        
        self.max_gops = max_gops
        self.dim_ctx = dim_ctx
        self.dim_mot = dim_mot
        self.fused_dim = self.dim_ctx + self.dim_mot

        # Temporal Decoder
        self.temporal_transformer = TemporalTransformer(
            d_model=self.fused_dim,
            n_layers=num_temporal_layers,
            n_head=n_head,
            max_gops=self.max_gops
        )

        # Classification Head
        self.head = ClassificationHead(
            input_dim = self.fused_dim,
            num_classes = num_classes
        )

    def forward(Self, iframes, mvs):
        B,N,C,H,W = iframes.shape
        _, _, C_m, T_m , H_m, W_m = mvs.shape

        # flatten 
        iframes_flat = iframes.view(B*N, C, H, W)
        mvs_flat = mvs.view(B*N, C_m, T_m, H_m, W_m)

        # encode
        ctx_feat = self.ctx_encoder(iframes_flat)
        mot_feat = self.mot_encoder(mvs_flat)

        # concatenate features
        fused_feat = torch.cat([ctx_feat, mot_feat], dim=1)
        
        # reshape back to temporal sequence
        sequence_feat = fused_feat.view(B,N, self.fused_dim)

        # temporal transformer
        video_feat = self.temporal_transformer(sequence_feat)

        # classification
        logits = self.head(video_feat)

        return logits