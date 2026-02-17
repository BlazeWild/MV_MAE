import numpy as np
import os
import av
import shutil

class RobustVideoPartitioner:
    def __init__(self, video_path, output_folder="gops", gop_size=17):
        self.video_path = video_path
        self.output_folder = output_folder
        self.gop_size = gop_size

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

    def rasterize_mvs(self, mvs, height, width):
        # Create a grid of shape (H/16, W/16, 2)
        mb_size = 16
        grid_h = height // mb_size
        grid_w = width // mb_size
        motion_field = np.zeros((grid_h, grid_w, 2), dtype=np.float32)

        for mv in mvs:
            # Get block coordinates
            bx = int(mv.dst_x // mb_size)
            by = int(mv.dst_y // mb_size)
            
            if 0 <= bx < grid_w and 0 <= by < grid_h:
                # Handle scale (sometimes 0 in headers)
                scale = mv.motion_scale if mv.motion_scale > 0 else 1
                # PyAV returns integer values, we need float pixels
                dx = mv.motion_x / scale
                dy = mv.motion_y / scale
                
                # Simple overwrite logic (fastest)
                motion_field[by, bx, 0] = dx
                motion_field[by, bx, 1] = dy
        return motion_field

    def process(self):
        # OPTION 1: Try setting it in the container options
        container = av.open(self.video_path, options={'export_mvs': 'true'})
        stream = container.streams.video[0]
        
        # OPTION 2: Manually force the codec context flag (For older/newer PyAV versions)
        try:
            # AV_CODEC_FLAG2_EXPORT_MVS = 1 << 28 (268435456)
            stream.codec_context.flags2 |= 268435456
        except AttributeError:
            pass # Some versions don't allow direct property access

        height = stream.codec_context.height
        width = stream.codec_context.width
        
        print(f"Processing {width}x{height} video...")

        gop_id = 0
        current_gop_frames = []
        
        # We track if we ever found a non-zero MV
        found_motion = False

        for frame in container.decode(stream):
            # 1. Get Motion Vectors
            mvs = frame.side_data.get('MOTION_VECTORS', [])
            
            if len(mvs) > 0:
                found_motion = True
                
            mv_data = self.rasterize_mvs(mvs, height, width)

            # 2. Get Image (Only for first frame of GOP)
            if len(current_gop_frames) == 0:
                img_data = frame.to_ndarray(format='rgb24')
            else:
                img_data = None 

            current_gop_frames.append({'image': img_data, 'motion': mv_data})

            # 3. Save if full
            if len(current_gop_frames) >= self.gop_size:
                self._save_gop(gop_id, current_gop_frames)
                gop_id += 1
                current_gop_frames = []

        # Flush
        if len(current_gop_frames) > 0:
            self._save_gop(gop_id, current_gop_frames)

        if not found_motion:
            print("\nWARNING: No Motion Vectors were found! Your output is all zeros.")
            print("Reason: Your video might be 'All-Intra' (frames are independent).")
            print("Fix: Re-encode it: ffmpeg -i input.mp4 output.mp4")

    def _save_gop(self, gop_id, frames):
        folder_name = os.path.join(self.output_folder, f"gop_{gop_id:04d}")
        os.makedirs(folder_name, exist_ok=True)
        
        iframe = frames[0]['image']
        np.save(os.path.join(folder_name, "iframe_rgb.npy"), iframe)

        # Stack shape: (16, H/16, W/16, 2)
        mvs = np.array([f['motion'] for f in frames])
        np.save(os.path.join(folder_name, "motion_vectors.npy"), mvs)
        
        if gop_id % 10 == 0:
            print(f"Saved GOP {gop_id} | MV Shape: {mvs.shape}")

if __name__ == "__main__":
    # Update this path
    part = RobustVideoPartitioner("training_fixed.mp4")
    part.process()