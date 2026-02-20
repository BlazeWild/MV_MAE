import torch
import torch.nn as nn
from .encoders import ContextEncoder, MotionEncoder
from .fusion import SimpleFusion

class MVMAE(nn.Module):
  """
  MV-MAE (Green Eye) Complete Architecture
  Connects the I-frane ecnoder , Motion and Fusion MOdel
  """
  
  def __init__(self, num_classes=155, model_zoo_path:str="./model_zoo", fusion_hidden_dim=256, droput=0.3):
    super().__init__()
    
    self.context_stream = COntextEncoder(model_zoo_path=model_zoo_path)
    self.motion_stream = MotionEncoder(model_zoo_path=model_zoo_path)
    
    self.fusion_head = SimpleFusion(
      dim_ctx=192,
      dim_mot=384,
      hidden_dim=fusion_hidden_dim,
      num_classes=num_classes,
      dropout=dropout
    )

  def forward(self, iframe, mvs):
    """
    The main forward pass:
    Args:
        i-frame(Tensor): [B, 3, 224, 224]
        mvs(Tensor): [B, 2, 16, H, W] ->H,W are resized to 224, 224
    Returns:
        logits(Tensor): [B, num_classes]
    """
    ctx_feat = self.context_stream(iframe)
    mot_feat = self.motion_stream(mvs)
    logits = self.fusion_head(ctx_feat, mot_feat)
    return logits