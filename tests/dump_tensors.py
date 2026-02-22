import os
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mv_mae.data.video_loader import VideoLoader

loader = VideoLoader("tests/training_fixed_480p.mp4", gop_size=16)
iframes_stack, mvs_stack = loader.get_video_clip(num_segments=8)

os.makedirs("tests/tensor_dump", exist_ok=True)
np.save("tests/tensor_dump/iframes.npy", iframes_stack.numpy())
np.save("tests/tensor_dump/mvs.npy", mvs_stack.numpy())
print("Tensors saved to tests/tensor_dump/")
