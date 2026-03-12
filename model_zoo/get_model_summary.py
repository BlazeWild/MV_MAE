import torch
import os
import timm
from torchinfo import summary

def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), "clip_context", "vit_base_patch16_224.pth")

    # True ViT-Base: 12 blocks, 768 dim, mlp_ratio=4
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict):
            if 'model' in state: state = state['model']
            elif 'model_state_dict' in state: state = state['model_state_dict']
        state = {k: v for k, v in state.items() if not k.startswith("head.")}
        model.load_state_dict(state, strict=True)
        print("Weights loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}")

    dummy_input = torch.randn(1, 3, 224, 224)
    summary(model, input_data=dummy_input, depth=3)

if __name__ == "__main__":
    main()
