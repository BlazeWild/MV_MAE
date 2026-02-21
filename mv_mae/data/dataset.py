import os
import random
import logging
import torch
from torch.utils.data import Dataset
from .video_loader import VideoLoader

logger = logging.getLogger(__name__)

class UAVHumanDataset(Dataset):
    def __init__(self, data_root, split_file, num_segments=8, gop_size=16, transform=None):
        """
        Args:
            data_root (str): Path to the folder containing the optimized .mp4 videos.
            split_file (str): Path to the generated train_split.txt or val_split.txt.
            num_segments (int): Number of temporal GOPs to sample (N).
            gop_size (int): Fixed frame size per GOP (hardware aligned to 16).
        """
        self.data_root = data_root
        self.split_file = split_file
        self.num_segments = num_segments
        self.gop_size = gop_size
        self.transform = transform
        
        self.samples = []
        self._load_split()

    def _load_split(self):
        """
        Reads the pre-generated text file mapping videos to labels.
        """
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
            
        logger.info(f"Loading dataset split from {self.split_file}...")
        
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                vid_name, label = parts[0], int(parts[1])
                vid_path = os.path.join(self.data_root, vid_name)
                
                # We only add files that actually exist to prevent I/O crashes
                if os.path.exists(vid_path):
                    self.samples.append((vid_path, label))
                
        logger.info(f"Successfully loaded {len(self.samples)} valid video samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path, label = self.samples[idx]
        
        try:
            loader = VideoLoader(vid_path, gop_size=self.gop_size)
            iframes, mvs = loader.get_video_clip(num_segments=self.num_segments)
            
            if iframes is None or mvs is None:
                raise ValueError("VideoLoader returned None (corrupt or unreadable file).")

            if self.transform:
                iframes, mvs = self.transform(iframes, mvs)

            # Cast label to long tensor for CrossEntropyLoss
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return iframes, mvs, label_tensor

        except Exception as e:
            # Major Project Failsafe: Log the corrupt file and grab a random substitute 
            # to keep the GPU training loop alive without raising a fatal error.
            logger.error(f"Error loading {vid_path}: {e}. Falling back to random sample.")
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)