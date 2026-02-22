import os
import numpy as np
import cv2

def reconstruct_from_arrays(gop_size=16):
    input_iframes = "tests/tensor_dump/iframes.npy"
    input_mvs = "tests/tensor_dump/mvs.npy"
    output_video = "tests/reconstruction_480p_from_npy.mp4"

    iframes_stack = np.load(input_iframes) # (8, 3, 480, 854)
    mvs_stack = np.load(input_mvs) # (8, 2, gop_size, H/16, W/16)

    N, C, H, W = iframes_stack.shape
    fps = 24.0 # Typical 112 frames at 24fps -> 5 seconds.
               # Let's change the output framerate to 24fps so 128 frames plays back in ~5.3 seconds.
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    for n in range(N):
        print(f"Looping GOP {n}")
        iframe_t = iframes_stack[n] # (3, H, W)
        mvs_t = mvs_stack[n]        # (2, 16, H/16, W/16)

        # Convert iframe from [0, 1] RGB tensor-like to [0, 255] BGR image
        iframe_np = (iframe_t.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        prev_frame = cv2.cvtColor(iframe_np, cv2.COLOR_RGB2BGR)

        out.write(prev_frame)

        for t in range(1, gop_size):
            # [2, H/16, W/16] -> [H/16, W/16, 2]
            mv_grid_t = mvs_t[:, t, :, :]
            mv_grid = mv_grid_t.transpose(1, 2, 0)

            flow = cv2.resize(mv_grid, (W, H), interpolation=cv2.INTER_NEAREST)

            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]

            curr_frame = cv2.remap(prev_frame, map_x, map_y, 
                                   interpolation=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REPLICATE)
            
            out.write(curr_frame)
            prev_frame = curr_frame

    out.release()
    print("Done generating 128 frames!")

if __name__ == '__main__':
    reconstruct_from_arrays()
