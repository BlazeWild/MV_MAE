import os
import sys
import torch
from torchinfo import summary

# Add the root repository folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pretraining.models.model_wrapper import DiscreteMVMAE

def check_checkpoint(checkpoint_path):
    print(f"Checking checkpoint: {checkpoint_path}")
    
    # Initialize model
    # Assuming default codebook_size of 1024 as seen in Config and model_wrapper
    model = DiscreteMVMAE(codebook_size=1024)
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # Load checkpoint
    try:
        # Using map_location='cpu' to be safe regardless of GPU availability
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Determine if it's a full training state or just the model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded full checkpoint from epoch: {epoch}")
        else:
            state_dict = checkpoint
            print("Loaded model state dict directly.")
            
        # Load state dict into model
        msg = model.load_state_dict(state_dict)
        print(f"State dict loaded with message: {msg}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Standard loading failed. If this is a .zip archive from gdown, you might need to unzip it first or PyTorch might handle it if it's the new format.")
        return

    # Model Summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    # Define dummy input sizes based on pretrain.py expectations: [B, 8, 2, 16, 14, 14]
    # We pass it as a list of inputs because forward(x, target_ids) requires two positional args
    batch_size = 2
    x = torch.randn(batch_size, 8, 2, 16, 14, 14)
    target_ids = torch.zeros(batch_size * 8, 1568).long()
    
    summary(model, input_data=[x, target_ids], depth=3, col_names=["input_size", "output_size", "num_params", "trainable"])

if __name__ == "__main__":
    # Path to the extracted checkpoint
    checkpoint_path = "/home/blaze/Documents/Windows_Backup/Ashok/_AI/_COMPUTER_VISION/____RESEARCH/MV_MAE/pretraining/checkpoints_pretrain/dmvmae_272_epoch_pretrain/latest_checkpoint.pth"
    
    check_checkpoint(checkpoint_path)
