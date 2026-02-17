import numpy as np
import cv2
import os
import glob

def visualize_gops(gop_folder="gops", output_video="reconstruction_debug.mp4"):
    # Get all GOP folders
    gop_dirs = sorted(glob.glob(os.path.join(gop_folder, "gop_*")))
    
    if not gop_dirs:
        print("No GOP folders found!")
        return

    # Read first GOP to get dimensions
    first_iframe = np.load(os.path.join(gop_dirs[0], "iframe_rgb.npy"))
    height, width, _ = first_iframe.shape
    
    # Setup Video Writer
    # Setup Video Writer
    # We will show Motion Visualization overlayed on I-Frame
    # The output width is original width.
    fps = 24.0 # Match source video FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Reconstructing video to {output_video}...")

    for gop_dir in gop_dirs:
        # Load Data
        iframe = np.load(os.path.join(gop_dir, "iframe_rgb.npy")) # (H, W, 3) RGB
        mvs = np.load(os.path.join(gop_dir, "motion_vectors.npy")) # (T, GridH, GridW, 2)

        # Convert I-Frame to BGR for OpenCV
        prev_frame = cv2.cvtColor(iframe, cv2.COLOR_RGB2BGR)
        
        # Write the first frame (I-Frame)
        out.write(prev_frame)

        # Generate coordinate grid once
        h, w = height, width
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_x = grid_x.astype(np.float32)
        grid_y = grid_y.astype(np.float32)

        # Iterate from t=1 (since t=0 is the I-frame itself)
        for t in range(1, mvs.shape[0]):
            mv_grid = mvs[t] # (GridH, GridW, 2)
            
            # Upscale Motion Vectors to pixel resolution
            # INTER_NEAREST preserves the blocky nature of standard MVs
            flow = cv2.resize(mv_grid, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Motion Compensation:
            # dst(x,y) = src(x + dx, y + dy)
            # The MV (dx, dy) points TO the position in the reference frame.
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            
            # Remap
            curr_frame = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            out.write(curr_frame)
            
            # Update for next frame (assume IPPPP structure, so next P uses this P as ref)
            prev_frame = curr_frame

    out.release()
    print("Done! Open reconstruction_debug.mp4 to see the reconstruction.")

if __name__ == "__main__":
    visualize_gops()