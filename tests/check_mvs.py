import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mv_mae.data.video_loader import VideoLoader

loader = VideoLoader("tests/training_fixed_480p.mp4", gop_size=16)
iframes, mvs = loader.get_video_clip(num_segments=8)

for g in range(8):
    for t in range(16):
        mv_t = mvs[g, :, t, :, :]
        non_zero = torch.count_nonzero(mv_t).item()
        print(f"GOP {g:02d} P-Frame {t:02d}: {non_zero:04d} non-zero MVs")
