import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
import faiss
import logging
from tqdm import tqdm
from pretraining.data.dataset import UAVHumanDataset

from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodebookConfig:
    """Master configuration for codebook generation."""
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.data_root = os.path.abspath(os.path.join(base_dir, '../../datasets/UAVHuman_480p_mp4/Action_Videos'))
        self.train_split = os.path.abspath(os.path.join(base_dir, '../../datasets/UAVHuman_480p_mp4/train_split.txt'))
        self.save_path = os.path.abspath(os.path.join(base_dir, '../../checkpoints/mv_codebook_1024.pt'))
        
        self.vocab_size = 1024
        self.gop_size = 16
        self.num_segments = 8
        
        # Memory & Sparsity Controls (Optimized for L4 GPU)
        self.max_videos_to_scan = 2000 
        self.max_vectors_to_keep = 5_000_000  # Caps RAM usage to ~40MB
        self.min_movement_threshold = 0.5     # Filters out static background


class MotionVectorExtractor:
    """Handles loading videos and extracting active, non-zero motion vectors using Multiprocessing."""
    def __init__(self, config):
        self.config = config
        self.dataset = self._initialize_dataset()

    def _initialize_dataset(self):
        logger.info("Initializing UAVHumanDataset...")
        return UAVHumanDataset(
            data_root=self.config.data_root, 
            split_file=self.config.train_split, 
            num_segments=self.config.num_segments, 
            gop_size=self.config.gop_size,
            is_train=True
        )

    def extract_active_vectors(self):
        """Extracts and filters vectors, returning a dense numpy array of active motion."""
        # Grab random indices for the subset
        indices = torch.randperm(len(self.dataset)).tolist()[:self.config.max_videos_to_scan]
        subset = Subset(self.dataset, indices)
        
        # THE FIX: Use PyTorch DataLoader for multiprocessing!
        # Batch size 8 + 10 workers means it processes chunks of videos in parallel across your vCPUs
        loader = DataLoader(
            subset, 
            batch_size=8, 
            shuffle=False, 
            num_workers=10,  # Leaving 2 vCPUs free for the main thread and OS
            pin_memory=False,
            drop_last=False
        )
        
        # Pre-allocate memory to prevent RAM fragmentation
        massive_pool = np.zeros((self.config.max_vectors_to_keep, 2), dtype=np.float32)
        current_count = 0
        
        logger.info(f"Scanning up to {len(indices)} videos using 10 CPU workers...")
        
        # Iterate over the parallelized dataloader
        for iframes, mvs, _ in tqdm(loader, desc="Extracting Batched MVs"):
            if current_count >= self.config.max_vectors_to_keep:
                break
                
            try:
                # mvs shape from dataloader: [Batch, Segments, Channels, Time, H, W]
                # Example: [8, 8, 2, 16, 14, 14]
                
                # Rearrange to put the 2 channels at the very end
                # Permute: (Batch, Segments, Time, H, W, Channels)
                mvs = mvs.permute(0, 1, 3, 4, 5, 2) 
                
                # Flatten everything except the 2 channels
                mvs_flat = mvs.reshape(-1, 2)    
                
                # Sparsity Filtering: Keep only vectors with actual movement
                mags = torch.linalg.norm(mvs_flat, dim=1)
                active_mvs = mvs_flat[mags > self.config.min_movement_threshold].numpy()
                
                num_active = active_mvs.shape[0]
                if num_active == 0:
                    continue
                    
                # Safely insert into the pre-allocated pool
                space_left = self.config.max_vectors_to_keep - current_count
                take = min(num_active, space_left)
                
                massive_pool[current_count : current_count + take] = active_mvs[:take]
                current_count += take
                
            except Exception as e:
                logger.warning(f"Batch processing error: {e}")
                continue
                
        logger.info(f"Successfully extracted {current_count:,} active motion vectors.")
        return massive_pool[:current_count]

class FaissClusterer:
    """Wraps the FAISS K-Means algorithm."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.d = 2 # Vector dimension (dx, dy)

    def train(self, data):
        """Trains the K-Means algorithm and returns the centroids as a PyTorch tensor."""
        logger.info(f"Launching FAISS K-Means (K={self.vocab_size}) on CPU (Bypassing cuBLAS bug)...")
        
        # THE FIX: Set gpu=False. 
        # FAISS will automatically use all your CPU threads via OpenMP.
        kmeans = faiss.Kmeans(
            d=self.d, 
            k=self.vocab_size, 
            niter=30,  
            verbose=True, 
            gpu=False   # <--- FIX IS HERE
        )
        
        kmeans.train(data)
        centroids = kmeans.centroids
        
        # Inject [0.0, 0.0] into Token 0 for padding/static masking
        centroids[0] = np.array([0.0, 0.0], dtype=np.float32)
        
        return torch.from_numpy(centroids)


class CodebookPipeline:
    """Master controller that executes the extraction and clustering process."""
    def __init__(self):
        self.config = CodebookConfig()
        self.extractor = MotionVectorExtractor(self.config)
        self.clusterer = FaissClusterer(self.config.vocab_size)
        
        os.makedirs(os.path.dirname(self.config.save_path), exist_ok=True)

    def run(self):
        # 1. Extract the data
        training_data = self.extractor.extract_active_vectors()
        
        # 2. Cluster the data
        centroids_tensor = self.clusterer.train(training_data)
        
        # 3. Save the results
        torch.save(centroids_tensor, self.config.save_path)
        logger.info(f"Codebook successfully saved to: {self.config.save_path}")
        logger.info(f"Final tensor shape: {centroids_tensor.shape}")


if __name__ == "__main__":
    pipeline = CodebookPipeline()
    pipeline.run()