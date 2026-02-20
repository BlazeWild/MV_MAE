import torch
import torch.nn as nn

class QUickGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

class SimpleFusion(nn.Module):
    """
    Concatenate the features and process them through an MLP
    Input: COntext feat [B, 192] and Motion Feat [B, 384]
    Output: Class Logits [B, Num_Classes]
    """
    def __init__(self, dim_ctx:int = 192, dim_mot:int = 384, hidden_dim:int = 256, num_classes:int = 155, dropout=0.2):
        super().__init__()
        
        input_dim = dim_ctx + dim_root

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.__init__weights()

    def __init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(Self, ctx_fea, mot_feat):
        fused = torch.cat([ctx_feat, mot_feat], dim =1)

        # pass through MLP to get logits
        logits = self.mlp(fused)
        return logits

