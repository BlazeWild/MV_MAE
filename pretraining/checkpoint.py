import os
import glob
import torch

def get_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file in the given directory."""
    # First check if latest_checkpoint.pth exists
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_path):
        return latest_path
        
    # Fallback to look for files matching the pattern mvmae_epoch_*.pth
    search_pattern = os.path.join(checkpoint_dir, "mvmae_epoch_*.pth")
    checkpoints = glob.glob(search_pattern)
    
    if not checkpoints:
        return None
    
    # Sort by the integer epoch number in the filename
    def extract_epoch(filepath):
        basename = os.path.basename(filepath)
        # e.g., mvmae_epoch_10.pth -> 10
        try:
            return int(basename.split('_')[-1].split('.')[0])
        except ValueError:
            return -1

    checkpoints.sort(key=extract_epoch)
    return checkpoints[-1]

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, logger=None):
    """Loads the model, optimizer, and scheduler states from a checkpoint."""
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # use weights_only=False since we load optimizer and scheduler states which might not be strictly tensors
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    
    if logger:
        logger.info(f"Successfully loaded checkpoint at epoch {epoch}")
        
    return epoch
