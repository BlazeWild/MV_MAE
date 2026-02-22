import os
import random
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .video_loader import VideoLoader

logger = logging.getLogger(__name__)

class AlignedSpatialCrop:
    """
    Ensures I-frames are 224x224 and handles the complex reshaping required 
    to feed Motion Vectors into a pre-trained VideoMAE model.
    """
    def __init__(self, crop_size=224, is_train=True):
        self.crop_size = crop_size
        self.mv_crop_size = crop_size // 16 
        self.is_train = is_train

    def __call__(self, iframes, mvs):
        # iframes: [N, 3, H, W]
        # mvs:     [N, 2, 16, H/16, W/16]
        _, _, H, W = iframes.shape
        
        # Calculate how many 16-pixel "steps" we can take for our crop anchor
        max_y_steps = max(0, (H - self.crop_size) // 16)
        max_x_steps = max(0, (W - self.crop_size) // 16)
        
        if self.is_train:
            step_y = random.randint(0, max_y_steps)
            step_x = random.randint(0, max_x_steps)
        else:
            step_y = max_y_steps // 2
            step_x = max_x_steps // 2
            
        # 1. Crop the I-Frames exactly on 16-pixel boundaries
        y, x = step_y * 16, step_x * 16
        iframes_cropped = iframes[:, :, y:y+self.crop_size, x:x+self.crop_size]
        
        # 2. Crop the Motion Vectors correspondingly
        mv_y, mv_x = step_y, step_x
        mvs_cropped = mvs[:, :, :, mv_y:mv_y+self.mv_crop_size, mv_x:mv_x+self.mv_crop_size]
        
        # 3. Upsample MVs back to 224x224 using Nearest Neighbor
        N, C_m, T_m, h, w = mvs_cropped.shape
        mvs_flat = mvs_cropped.view(N, C_m * T_m, h, w)
        mvs_up = F.interpolate(mvs_flat, scale_factor=16, mode='nearest')
        mvs_up = mvs_up.view(N, C_m, T_m, self.crop_size, self.crop_size)
        
        # 4. Pad a 3rd channel (Zeros) to match RGB pre-training shape
        zeros = torch.zeros((N, 1, T_m, self.crop_size, self.crop_size), dtype=mvs_up.dtype, device=mvs_up.device)
        mvs_3ch = torch.cat([mvs_up, zeros], dim=1) 
        
        return iframes_cropped, mvs_3ch


class UAVHumanDataset(Dataset):
    def __init__(self, data_root, split_file, num_segments=8, gop_size=16, is_train=True):
        self.data_root = data_root
        self.split_file = split_file
        self.num_segments = num_segments
        self.gop_size = gop_size
        
        self.cropper = AlignedSpatialCrop(crop_size=224, is_train=is_train)
        
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

    def __getitem__(self, idx):
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
            return self.__getitem__(random_idx)