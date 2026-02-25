#dataset.py
import os
import random
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .video_loader import VideoLoader

logger = logging.getLogger(__name__)

class UAVDataTransform:
    """
    Ensures I-frames are 224x224 and handles the complex reshaping required 
    to feed Motion Vectors into a pre-trained VideoMAE model.
    """
    def __init__(self, crop_size=224, is_train=True):
        self.crop_size = crop_size
        self.mv_crop_size = crop_size // 16  # 224/16 = 14
        self.is_train = is_train

    def __call__(self, iframes, mvs):
        # iframes: [N, 3, H, W]
        # mvs : [N, 2, 16, H_grid, W_grid]
        
        N, C_i, H_i, W_i = iframes.shape
        N, C_m, T_m, H_grid, W_grid = mvs.shape    

        # 1. I-frame Processing: Resize to 224x224
        iframes_resized = F.interpolate(iframes, size=(self.crop_size, self.crop_size), mode='bilinear', align_corners=False)

        # 2. MV Camera Ego Motion Compensation (Median Subtraction)
        mvs_flat = mvs.view(N, C_m, T_m, -1)  # [N, 2, 16, H_grid*W_grid]
        
        # Calculate median
        medians, _ = torch.median(mvs_flat, dim=-1, keepdim=True)  # [N, 2, 16, 1]
        medians = medians.unsqueeze(-1)                            # [N, 2, 16, 1, 1]
        
        # Subtract median drone movement
        mvs_compensated = mvs - medians  # [N, 2, 16, H_grid, W_grid]

        # 3. MV Adaptive Average Pooling (Crush to 14x14)
        # We pack Batch (N) and Time (T_m) together, but leave Channels (C_m) intact!
        mvs_packed = mvs_compensated.view(N * T_m, C_m, H_grid, W_grid)  # [128, 2, H_grid, W_grid]
        
        # Smoothly pool to 14x14
        mvs_pooled = F.adaptive_avg_pool2d(mvs_packed, (self.mv_crop_size, self.mv_crop_size))  # [128, 2, 14, 14]
        
        # UNPACK back to 5D tensor shape required by VideoMAE
        mvs_final = mvs_pooled.view(N, C_m, T_m, self.mv_crop_size, self.mv_crop_size)  # [8, 2, 16, 14, 14]

        return iframes_resized, mvs_final


class UAVHumanDataset(Dataset):
    def __init__(self, data_root, split_file, num_segments=8, gop_size=16, is_train=True):
        self.data_root = data_root
        self.split_file = split_file
        self.num_segments = num_segments
        self.gop_size = gop_size
        
        self.cropper = UAVDataTransform(crop_size=224, is_train=is_train)
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.samples = []
        self._load_split()

    def _load_split(self):
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
            
        with open(self.split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    vid_path = os.path.join(self.data_root, parts[0])
                    if os.path.exists(vid_path):
                        self.samples.append((vid_path, int(parts[1])))
                        
        logger.info(f"Successfully loaded {len(self.samples)} valid video samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, retries=0):
        if retries > 10:
            raise RuntimeError("Too many corrupted files in sequence, DataLoader aborting.")

        vid_path, label = self.samples[idx]
        
        try:
            loader = VideoLoader(vid_path, gop_size=self.gop_size)
            iframes, mvs = loader.get_video_clip(num_segments=self.num_segments)
            
            if iframes is None or mvs is None:
                raise ValueError("VideoLoader returned None (stream unreadable).")

            iframes, mvs = self.cropper(iframes, mvs)
            iframes = (iframes - self.mean) / self.std

            label_tensor = torch.tensor(label, dtype=torch.long)
            return iframes, mvs, label_tensor

        except Exception as e:
            logger.warning(f"Error loading {vid_path}: {e}. Trying random sample.")
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx, retries=retries+1)