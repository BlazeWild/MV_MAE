import torch
import os
from pathlib import Path

def get_params_from_checkpoint(ckpt_path):
    """
    Loads a checkpoint file (.bin or .pth) and calculates the number of parameters
    directly from the state_dict, without instantiating the model class.
    """
    try:
        # Load state dict
        # map_location='cpu' ensures we don't need a GPU
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # Handle cases where state_dict is nested (e.g., {'model': ...})
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        total_params = 0
        trainable_params = 0 # Cannot strictly determine without model, assuming all in state_dict are params
        
        print(f"\nAnalyzing: {os.path.basename(ckpt_path)}")
        print("-" * 40)
        
        for name, param in state_dict.items():
            # Skip non-parameter entries if any (like batch norm stats sometimes saved distinctly, 
            # though usually they are tensors in state_dict)
            if isinstance(param, torch.Tensor):
                num_params = param.numel()
                total_params += num_params
                # print(f"{name}: {num_params}") # Uncomment for detailed layer-wise count

        # Convert to Millions
        params_m = total_params / 1e6
        
        print(f"Total Parameters: {total_params:,} ({params_m:.2f} M)")
        return total_params

    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return 0

def main():
    # Base directory for model zoo
    base_dir = Path(__file__).parent
    
    # Find all .bin or .pth files in subdirectories
    model_files = list(base_dir.rglob("*.bin")) + list(base_dir.rglob("*.pth"))
    
    if not model_files:
        print(f"No .bin or .pth files found in {base_dir}")
        return

    print(f"Found {len(model_files)} model files.")
    
    for model_file in model_files:
         get_params_from_checkpoint(model_file)

if __name__ == "__main__":
    main()
