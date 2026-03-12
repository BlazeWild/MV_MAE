import torch 
import torch.nn as nn
import logging
from .encoders import ContextEncoder, MotionEncoder
from .temporal import TemporalTransformer

logger = logging.getLogger(__name__)

class ClassificationHead(nn.Module):
    """Takes the final video-level feature and outputs class predictions."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 155, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class MVMAE(nn.Module):
    """MV-MAE Complete Hierarchical Architecture"""
    def __init__(self, num_classes=155, model_zoo_path="./model_zoo", pretrain_ckpt_path=None, d_model=512, num_temporal_layers=4, n_head=8, dim_ctx=768, dim_mot=384, max_gops=8):
        super().__init__()
        # Spatial backbone (I-frame encoder)
        self.ctx_encoder = ContextEncoder(model_zoo_path)
        # Motion backbone (pretrained MVMAEEncoder)
        self.mot_encoder = MotionEncoder(pretrain_ckpt_path)
        
        self.max_gops = max_gops
        self.dim_ctx = dim_ctx
        self.dim_mot = dim_mot
        self.raw_fused_dim = self.dim_ctx + self.dim_mot  # 1152
        self.d_model = d_model                            # 512

        # Fusion Projection: compress concatenated features before temporal modeling
        self.fusion_projection = nn.Linear(self.raw_fused_dim, self.d_model)  # 1152 -> 512

        # Temporal Decoder (4 layers, high dropout to prevent overfitting on short 8-GOP sequence)
        self.temporal_transformer = TemporalTransformer(
            d_model=self.d_model,
            n_layers=num_temporal_layers,
            n_head=n_head,
            max_gops=self.max_gops,
            dropout=0.3
        )

        # Classification Head
        self.head = ClassificationHead(
            input_dim=self.d_model,
            num_classes=num_classes
        )

    # ==================================================================
    # 2-Phase SFT: Freeze / Unfreeze Methods
    # ==================================================================
    def freeze_encoders(self):
        """Phase 1: Freeze ALL parameters in both ViT encoders."""
        frozen_count = 0
        for name, param in self.ctx_encoder.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        for name, param in self.mot_encoder.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        
        self.ctx_encoder.eval()
        self.mot_encoder.eval()
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[Phase 1] Froze {frozen_count} encoder parameter tensors.")
        logger.info(f"[Phase 1] Trainable params remaining: {trainable:,}")

    def unfreeze_encoder_top_layers(self, ctx_top_n=2, mot_top_n=2):
        """Phase 2: Unfreeze the top N transformer blocks of each encoder."""
        unfrozen_count = 0
        
        # --- ViT-Base (ContextEncoder) ---
        # Unfreeze top ctx_top_n blocks from self.ctx_encoder.vit.blocks
        vit_blocks = list(self.ctx_encoder.vit.blocks)
        total_ctx = len(vit_blocks)
        for block in vit_blocks[total_ctx - ctx_top_n:]:
            for param in block.parameters():
                param.requires_grad = True
                unfrozen_count += 1
        # Also unfreeze ViT's final norm
        if hasattr(self.ctx_encoder.vit, 'norm'):
            for param in self.ctx_encoder.vit.norm.parameters():
                param.requires_grad = True
                unfrozen_count += 1
        
        # --- MVMAEEncoder (MotionEncoder) ---
        # Unfreeze top mot_top_n blocks from self.mot_encoder.encoder.blocks
        mot_blocks = list(self.mot_encoder.encoder.blocks)
        total_mot = len(mot_blocks)
        for block in mot_blocks[total_mot - mot_top_n:]:
            for param in block.parameters():
                param.requires_grad = True
                unfrozen_count += 1
        # Also unfreeze encoder's final norm
        if hasattr(self.mot_encoder.encoder, 'norm'):
            for param in self.mot_encoder.encoder.norm.parameters():
                param.requires_grad = True
                unfrozen_count += 1
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[Phase 2] Unfroze {unfrozen_count} encoder parameter tensors "
                     f"(ViT-Base top {ctx_top_n}/{total_ctx}, MVMAEEncoder top {mot_top_n}/{total_mot}).")
        logger.info(f"[Phase 2] Total trainable params: {trainable:,}")

    def get_param_groups(self, lr_head, lr_encoder, weight_decay=0.05):
        """
        Returns optimizer param groups with separate LRs:
          - Group 1 (lr_head):    TemporalTransformer + ClassificationHead 
          - Group 2 (lr_encoder): Unfrozen encoder params (top layers only)
        """
        head_decay, head_no_decay = [], []
        encoder_decay, encoder_no_decay = [], []
        
        # Fusion Projection + Temporal Transformer + Classification Head params
        for module in [self.fusion_projection, self.temporal_transformer, self.head]:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)
        
        # Unfrozen encoder params (requires_grad=True in the encoder modules)
        for module in [self.ctx_encoder, self.mot_encoder]:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    encoder_no_decay.append(param)
                else:
                    encoder_decay.append(param)
        
        param_groups = [
            {"params": head_decay,       "lr": lr_head,    "weight_decay": weight_decay},
            {"params": head_no_decay,    "lr": lr_head,    "weight_decay": 0.0},
        ]
        
        # Only add encoder groups if there are unfrozen encoder params
        if encoder_decay or encoder_no_decay:
            param_groups.append(
                {"params": encoder_decay,    "lr": lr_encoder, "weight_decay": weight_decay}
            )
            param_groups.append(
                {"params": encoder_no_decay, "lr": lr_encoder, "weight_decay": 0.0}
            )
            logger.info(f"Optimizer: {len(head_decay)+len(head_no_decay)} head params (lr={lr_head}), "
                         f"{len(encoder_decay)+len(encoder_no_decay)} encoder params (lr={lr_encoder})")
        else:
            logger.info(f"Optimizer: {len(head_decay)+len(head_no_decay)} head params (lr={lr_head}), "
                         f"0 encoder params (all frozen)")
        
        return param_groups

    def forward(self, iframes, mvs):
        # iframes: [B, N, C, H, W]
        # mvs:     [B, N, C_m, T_m, H_m, W_m]
        B, N, C, H, W = iframes.shape
        _, _, C_m, T_m, H_m, W_m = mvs.shape

        # 1. Flatten the Batch and Sequence dimensions together for the independent encoders
        iframes_flat = iframes.view(B * N, C, H, W)
        mvs_flat = mvs.view(B * N, C_m, T_m, H_m, W_m)

        # 2. Extract features
        ctx_feat = self.ctx_encoder(iframes_flat) # Output: [B*N, 768]
        mot_feat = self.mot_encoder(mvs_flat)     # Output: [B*N, 384]

        # 3. Concatenate features and project to stable dimension
        fused_feat = torch.cat([ctx_feat, mot_feat], dim=1)   # [B*N, 1152]
        fused_feat = self.fusion_projection(fused_feat)        # [B*N, 512]
        
        # 4. Reshape back to chronological temporal sequence for the Transformer
        sequence_feat = fused_feat.view(B, N, self.d_model)    # [B, N, 512]

        # 5. Process through Temporal Transformer to get global video context
        video_feat = self.temporal_transformer(sequence_feat)   # [B, 512]

        # 6. Final Classification
        logits = self.head(video_feat)                          # [B, 155]

        return logits