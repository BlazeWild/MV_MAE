import torch
import sys
import os
from torchinfo import summary

# Ensure mv_mae is reachable to load the encoders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mv_mae.models.encoders import ContextEncoder, MotionEncoder
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Encoders. Error: {e}")
    MODELS_AVAILABLE = False

def print_model_summaries():
    """
    Prints the torchinfo hierarchical summary for the ContextEncoder and MotionEncoder.
    """
    if not MODELS_AVAILABLE:
        print("\nCould not load Encoders for torchinfo summary.")
        return

    base_dir = os.path.dirname(__file__)

    # 1. Context Encoder (ViT-Tiny)
    print("\n" + "="*80)
    print("ContextEncoder (ViT-Tiny) Summary")
    print("="*80)
    try:
        context_encoder = ContextEncoder(model_zoo_path=base_dir)
        iframe = torch.randn(1, 3, 224, 224)
        summary(context_encoder, input_data=[iframe], 
                col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
                depth=4, verbose=1)
    except Exception as e:
        print(f"Failed to load/summarize ContextEncoder: {e}")

    # 2. Motion Encoder (VideoMAE-Small)
    print("\n" + "="*80)
    print("MotionEncoder (VideoMAE-Small) Summary")
    print("="*80)
    try:
        motion_encoder = MotionEncoder(model_zoo_path=base_dir)
        mvs = torch.randn(1, 3, 16, 224, 224)
        summary(motion_encoder, input_data=[mvs], 
                col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
                depth=4, verbose=1)
    except Exception as e:
        print(f"Failed to load/summarize MotionEncoder: {e}")

if __name__ == "__main__":
    print_model_summaries()
