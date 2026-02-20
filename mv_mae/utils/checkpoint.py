import os
import glob
import shutil
import torch
import logging

logger = logging.getLogger(__name__)

def get_latest_checkpoint(ckpt_dir):
    """
    Automatically finds the most recently modified .pth checkpoint in the directory.
    """
    if not os.path.exists(ckpt_dir):
        return None
        
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
    if not ckpt_files:
        return None
        
    # Sort by modification time to get the absolute latest
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    return latest_ckpt


def save_checkpoint(state, is_best, ckpt_dir="./checkpoints", filename="checkpoint.pth"):
    """
    Saves the training state. If it's the best epoch, makes a separate 'best' copy.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    filepath = os.path.join(ckpt_dir, filename)
    
    # Save the current state
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint: {filepath}")
    
    # If it's the highest accuracy so far, save it as the 'best' model
    if is_best:
        best_filepath = os.path.join(ckpt_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)
        logger.info(f"--> Updated best model: {best_filepath}")


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Resumes training from a saved checkpoint, handling multi-GPU prefixes safely.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at '{ckpt_path}'")

    logger.info(f"Loading checkpoint '{ckpt_path}' to {device}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 1. Handle Multi-GPU (DDP) prefix mismatch
    # If the model was saved on multi-GPU but loaded on single-GPU (or vice versa)
    state_dict = checkpoint['model_state_dict']
    
    # Clean 'module.' prefix if it exists but the current model isn't wrapped
    if not hasattr(model, 'module'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # Add 'module.' prefix if current model IS wrapped but checkpoint wasn't
    elif hasattr(model, 'module') and not list(state_dict.keys())[0].startswith('module.'):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    # 2. Load Model Weights
    model.load_state_dict(state_dict, strict=True)
    
    # 3. Load Optimizer and Scheduler States
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Resumed optimizer state.")
        
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Resumed scheduler state.")
        
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    logger.info(f"Successfully loaded checkpoint (Resuming from Epoch {start_epoch} | Best Acc: {best_acc:.2f}%)")
    
    return start_epoch, best_acc