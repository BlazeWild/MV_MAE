import sys
import os
import subprocess
import torch
import numpy as np
import cv2

# Add parent dir to sys.path to import mv_mae module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mv_mae.data.video_loader import VideoLoader

def resize_video(input_path, output_path, target_height=480):
    print(f"Resizing {input_path} to {target_height}p -> {output_path}")
    if os.path.exists(output_path):
        print("Output file already exists, skipping resize...")
        return True
        
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", f"scale=-2:{target_height}",
        "-c:v", "libx264", "-g", "16", "-keyint_min", "16",
        "-c:a", "copy",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error resizing video: {e}")
        return False

def reconstruct_video(video_path, output_video, num_segments=8, gop_size=16):
    print(f"Loading GOPs from {video_path} using VideoLoader...")
    
    # We initialize the VideoLoader to read I-frames and MVs
    loader = VideoLoader(video_path, gop_size=gop_size)
    iframes_stack, mvs_stack = loader.get_video_clip(num_segments=num_segments)
    
    if iframes_stack is None:
        print("Failed to load video or extract GOPs.")
        return
        
    print(f"Loaded {num_segments} GOPs successfully.")
    print(f"IFrames shape: {iframes_stack.shape}")
    print(f"MVs shape:     {mvs_stack.shape}")
    
    # iframes_stack shape: [num_segments, 3, H, W]
    N, C, H, W = iframes_stack.shape
    
    fps = 24.0 # Matches typical UAVHuman fps or assumes standard
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    
    print(f"Writing reconstruction to {output_video}...")
    
    # Precompute grid_x and grid_y for the remap function
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    for n in range(N):
        iframe_t = iframes_stack[n] # [3, H, W]
        mvs_t = mvs_stack[n]        # [2, gop_size, H/16, W/16]
        
        # Convert I-Frame back to numpy RGB [0, 255] then to BGR for OpenCV
        iframe_np = (iframe_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        prev_frame = cv2.cvtColor(iframe_np, cv2.COLOR_RGB2BGR)
        
        # Write the I-Frame
        out.write(prev_frame)
        
        # Loop through P-frames in the GOP (index 1 to gop_size - 1)
        for t in range(1, gop_size):
            # Extract MV grid for time t: [2, H/16, W/16] -> [H/16, W/16, 2]
            mv_grid_t = mvs_t[:, t, :, :]
            mv_grid = mv_grid_t.permute(1, 2, 0).numpy()
            
            # Upscale Motion Vectors to full pixel resolution using Nearest Neighbor
            flow = cv2.resize(mv_grid, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Motion Compensation mapping: remap points to previous positions
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            
            curr_frame = cv2.remap(prev_frame, map_x, map_y, 
                                   interpolation=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            # Write P-frame and update prev_frame for the next iteration step
            out.write(curr_frame)
            prev_frame = curr_frame

    out.release()
    print("Done! Open it to see the visual reconstruction.")

if __name__ == "__main__":
    input_video = "tests/training_fixed.mp4"
    resized_video = "tests/training_fixed_480p.mp4"
    reconstruction_video = "tests/reconstruction_480p.mp4"
    
    # Standard values matching default settings
    num_segments = 8
    gop_size = 16 

    if resize_video(input_video, resized_video, target_height=480):
        reconstruct_video(resized_video, reconstruction_video, num_segments, gop_size)
