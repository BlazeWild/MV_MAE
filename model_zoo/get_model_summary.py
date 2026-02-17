import torch
import sys
import os
from torchinfo import summary

# Ensure mv_mae is reachable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mv_mae.models.mv_mae import MvMaeNET
    MODEL_AVAILABLE = True
except ImportError:
    print("Warning: Could not import MvMaeNET.")
    MODEL_AVAILABLE = False

def check_model_summary():
    """
    Checks the model parameters and structure using torchinfo.
    """
    if MODEL_AVAILABLE:
        # Load the model (Kinetics-400 size)
        model = MvMaeNET(num_classes=400)
        
        # Create Dummy Inputs
        # 1. I-Frame (Batch, 3, Height, Width)
        iframe = torch.randn(1, 3, 224, 224)
        
        # 2. Motion Vectors (Batch, 2, Frames, Height, Width)
        # Using 16 frames as default
        mvs = torch.randn(1, 2, 16, 224, 224)
        
        print("\n" + "="*50)
        print("Model Summary")
        print("="*50)
        
        # Use torchinfo to print detailed summary
        summary(model, input_data=[iframe, mvs], 
                col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
                depth=4, verbose=1)
    else:
        print("Model definition not found. Please verify the `mv_mae` package structure.")

if __name__ == "__main__":
    check_model_summary()
