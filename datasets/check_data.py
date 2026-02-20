import torch
from torch.utils.data import DataLoader
from data.dataset import UAVHumanDataset
from models.mv_mae import MVMAE

def test_pipeline():
    print("1. Initializing Dataset...")
    # NOTE: Point this to a dummy text file with just 5 or 10 videos to test
    dummy_dataset = UAVHumanDataset(
        data_root="./dummy_videos", 
        split_file="./dummy_split.txt", 
        num_segments=8
    )
    
    loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
    
    print("2. Fetching one batch...")
    iframes, mvs, labels = next(iter(loader))
    
    print(f"I-Frames shape: {iframes.shape}") # Should be [2, 8, 3, 224, 224]
    print(f"Motion Vec shape: {mvs.shape}")   # Should be [2, 8, 2, 16, 224, 224]
    print(f"Labels: {labels}")
    
    print("3. Passing through MV-MAE Architecture...")
    model = MVMAE(num_classes=155)
    
    # Forward pass
    logits = model(iframes, mvs)
    
    print(f"4. Output Logits shape: {logits.shape}") # Should be [2, 155]
    print("SUCCESS! The data flows perfectly.")

if __name__ == "__main__":
    test_pipeline()