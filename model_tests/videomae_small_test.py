import os
import torch
import numpy as np
import av
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_ROOT = '../datasets/UAVHuman_480p_mp4/Action_Videos'
TEST_SPLIT = '../datasets/UAVHuman_480p_mp4/test_split.txt'
MODEL_PATH = '../model_zoo/video_mae/vit_s_k400_ft.pth'
NUM_CLASSES = 155
NUM_FRAMES = 16
CROP_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==============================================================================
# DATASET AND VIDEO LOADER (RGB FRAMES)
# ==============================================================================
def read_video_pyav(container, indices):
    """
    Reads specific frame indices from a video using PyAV.
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_rgb().to_ndarray())
            
    return np.stack(frames)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Spreads requested frames evenly across the entire video.
    """
    converted_len = int(clip_len * frame_sample_rate)
    
    # If the video is shorter than what we need, just repeat the last frame 
    # Or start from 0 if it's long enough
    if converted_len > seg_len:
        converted_len = seg_len
    
    end_idx = np.random.randint(converted_len, seg_len) if seg_len > converted_len else seg_len - 1
    start_idx = end_idx - converted_len
    if start_idx < 0:
        start_idx = 0
        
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def preprocess_frames(frames_np):
    """
    frames_np: [T, H, W, C] numpy array
    Returns: [C, T, H, W] tensor normalized
    """
    import torchvision.transforms as T
    
    # Center crop and resize
    T_C, H, W, C = frames_np.shape
    
    # Simple PyTorch transforms 
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.CenterCrop((CROP_SIZE, CROP_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed = []
    for i in range(T_C):
        processed.append(transform(frames_np[i]))
        
    tensor_frames = torch.stack(processed, dim=1) # [C, T, H, W]
    
    return tensor_frames


class UAVHumanRGBDataset(Dataset):
    def __init__(self, data_root, split_file, num_frames=16):
        self.data_root = data_root
        self.num_frames = num_frames
        self.samples = []
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")
            
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    vid_path = os.path.join(self.data_root, parts[0])
                    label = int(parts[1])
                    if os.path.exists(vid_path):
                        self.samples.append((vid_path, label))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        vid_path, label = self.samples[idx]
        try:
            container = av.open(vid_path)
            total_frames = container.streams.video[0].frames
            
            # Simple fallback if total_frames isn't available
            if total_frames <= 0:
                total_frames = 100 
                
            indices = sample_frame_indices(clip_len=self.num_frames, frame_sample_rate=1, seg_len=total_frames)
            video = read_video_pyav(container, indices)
            
            # Preprocess to [C, T, H, W]
            tensor_video = preprocess_frames(video)
            
            # Need to swap channels for HuggingFace VideoMAE?
            # HF VideoMAE expects `pixel_values` of shape `(num_frames, num_channels, height, width)`
            hf_video = tensor_video.permute(1, 0, 2, 3) 
            
            return hf_video, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error reading {vid_path}: {e}")
            # Return dummy zeroes and dummy label to not crash the loader 
            return torch.zeros((self.num_frames, 3, CROP_SIZE, CROP_SIZE)), torch.tensor(-1, dtype=torch.long)

# ==============================================================================
# MAIN TEST SCRIPT
# ==============================================================================
def main():
    print("Initializing test dataset...")
    # Get absolute paths 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We navigate one level up from `model_tests` to `MV_MAE`
    proj_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    abs_data_root = os.path.join(proj_root, DATA_ROOT.replace('../', ''))
    abs_split_file = os.path.join(proj_root, TEST_SPLIT.replace('../', ''))
    abs_model_path = os.path.join(proj_root, MODEL_PATH.replace('../', ''))
    
    test_dataset = UAVHumanRGBDataset(
        data_root=abs_data_root,
        split_file=abs_split_file,
        num_frames=NUM_FRAMES
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Loaded {len(test_dataset)} test samples.")
    
    print("Loading VideoMAE Small Model for Classification...")
    model_dir = os.path.dirname(abs_model_path)
    
    config = VideoMAEConfig.from_pretrained(model_dir)
    config.num_labels = NUM_CLASSES
    
    # Initialize classification architecture
    model = VideoMAEForVideoClassification(config)
    
    # Load original weights. Strict=False because classifier head shape will be different
    print("Loading weights (strict=False) to allow classifier head replacement...")
    state_dict = torch.load(abs_model_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    # Remove head weights if shape mismatch occurs to prevent errors, and initialize with random weights
    if 'classifier.weight' in state_dict and state_dict['classifier.weight'].shape[0] != NUM_CLASSES:
        print("Mismatched classifier shape detected. Removing pretrained head weights...")
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']
        
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    try:
        from torchinfo import summary
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        summary(model, input_size=(BATCH_SIZE, NUM_FRAMES, 3, CROP_SIZE, CROP_SIZE))
        print("="*50 + "\n")
    except ImportError:
        print("torchinfo not installed. Skipping model summary.")
    
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            # Skip invalid videos
            valid_idx = labels != -1
            if not valid_idx.any():
                continue
                
            videos = videos[valid_idx].to(DEVICE)
            labels = labels[valid_idx].to(DEVICE)
            
            # HF model forward
            outputs = model(pixel_values=videos)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    # Calculate Overall Accuracy
    accuracy = (all_preds_np == all_labels_np).mean()

    print("\n" + "="*50)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print("="*50 + "\n")
    
    print("Saving confusion matrix and predictions...")
    
    # Custom Confusion Matrix implementation
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels_np, all_preds_np):
        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[t, p] += 1

    
    # Save raw CM data to NPY
    np.save(os.path.join(script_dir, "confusion_matrix.npy"), cm)
    
    # Save CM to CSV
    import csv
    csv_path = os.path.join(script_dir, "confusion_matrix.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(["True\\Pred"] + [str(i) for i in range(NUM_CLASSES)])
        for i in range(NUM_CLASSES):
            writer.writerow([str(i)] + cm[i].tolist())
            
    print(f"Results saved to {script_dir}/confusion_matrix.npy and .csv")

if __name__ == "__main__":
    main()
