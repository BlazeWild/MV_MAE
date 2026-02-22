import av
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class VideoLoader:
    def __init__(self, video_path, gop_size=16):
        """
        Extracts compressed domain data (I-Frames + Motion Vectors) efficiently.
        """
        self.video_path = video_path
        self.gop_size = gop_size
        self.width = 0
        self.height = 0

    def rasterize_mvs(self, mvs, height, width):
        """
        Converts sparse motion vectors into a dense, averaged motion field (H/16, W/16, 2).
        Handles overlapping vectors gracefully.
        """
        mb_size = 16
        grid_h = height // mb_size
        grid_w = width // mb_size
        
        motion_field = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
        counts = np.zeros((grid_h, grid_w, 1), dtype=np.float32)

        for mv in mvs:
            bx = int(mv.dst_x // mb_size)
            by = int(mv.dst_y // mb_size)
            
            if 0 <= bx < grid_w and 0 <= by < grid_h:
                scale = mv.motion_scale if mv.motion_scale > 0 else 1
                dx = mv.motion_x / scale
                dy = mv.motion_y / scale
                
                motion_field[by, bx, 0] += dx
                motion_field[by, bx, 1] += dy
                counts[by, bx, 0] += 1

        with np.errstate(invalid='ignore'):
            motion_field = np.divide(motion_field, counts, out=motion_field, where=counts>0)
            
        return motion_field

    def get_video_clip(self, num_segments=8):
        """
        Calculates 16-frame sub-GOPs via a chronological scan.
        This handles the CPU choke by not storing unused images, avoids PyAV seek() 
        errors on single-I-frame videos, and eliminates the Pseudo-I-frame MV trap.
        """
        container = av.open(self.video_path, options={'export_mvs': 'true'})
        stream = container.streams.video[0]
        
        try:
            stream.codec_context.flags2 |= 268435456  # AV_CODEC_FLAG2_EXPORT_MVS
        except AttributeError:
            pass

        self.height = stream.codec_context.height
        self.width = stream.codec_context.width

        all_gops = []
        current_gop_frames = []
        
        for frame in container.decode(stream):
            is_iframe = frame.key_frame
            
            # If a true I-frame appears and we have an incomplete GOP, save it (it gets post-padded later)
            if is_iframe and len(current_gop_frames) > 0:
                all_gops.append(current_gop_frames)
                current_gop_frames = []
                
            mvs = frame.side_data.get('MOTION_VECTORS', [])
            mv_data = self.rasterize_mvs(mvs, self.height, self.width)
            
            # Core Fixes:
            # 1. We ONLY decode the RGB image on the first frame of a chunk (saving CPU/RAM)
            # 2. We ZERO OUT the Motion Vector for the first frame of the chunk. 
            #    This prevents the "Pseudo-I-Frame Trap" where VideoMAE expects an anchor 
            #    but the P-frame tries to reference motion from prior undisplayed frames.
            if len(current_gop_frames) == 0:
                img_data = frame.to_ndarray(format='rgb24')
                mv_data = np.zeros_like(mv_data)
            else:
                img_data = None
                
            current_gop_frames.append({'image': img_data, 'motion': mv_data})
            
            # Hard reset the GOP when it naturally hits 16 frames
            if len(current_gop_frames) == self.gop_size:
                all_gops.append(current_gop_frames)
                current_gop_frames = []
                
        # Append whatever remains at the end of the file
        if len(current_gop_frames) > 0:
            all_gops.append(current_gop_frames)
            
        container.close()

        if not all_gops:
            return None, None

        # Uniformly sample num_segments (8) chunks from the total collected
        total_gops = len(all_gops)
        if total_gops >= num_segments:
            indices = np.linspace(0, total_gops - 1, num_segments).astype(int)
        else:
            indices = np.array([i % total_gops for i in range(num_segments)])

        sampled_gops = [all_gops[i] for i in indices]

        iframe_tensors = []
        mvs_tensors = []
        
        for gop_frames in sampled_gops:
            iframe_t, mvs_t = self._pack_and_pad_gop(gop_frames)
            if iframe_t is not None and mvs_t is not None:
                iframe_tensors.append(iframe_t)
                mvs_tensors.append(mvs_t)

        if len(iframe_tensors) != num_segments:
            return None, None

        return torch.stack(iframe_tensors), torch.stack(mvs_tensors)

    def _pack_and_pad_gop(self, frames):
        """
        Enforces a strict temporal dimension of `gop_size` using zero-motion post-padding.
        """
        actual_len = len(frames)
        
        if actual_len == 0 or frames[0]['image'] is None:
            return None, None
            
        if actual_len > self.gop_size:
            frames = frames[:self.gop_size]
        elif actual_len < self.gop_size:
            padding_needed = self.gop_size - actual_len
            zero_motion = np.zeros_like(frames[-1]['motion'])
            
            # Post-padding: Zeros are appended to the END of the sequence
            for _ in range(padding_needed):
                frames.append({'image': None, 'motion': zero_motion})

        iframe_np = frames[0]['image']
        iframe_tensor = torch.from_numpy(iframe_np).permute(2, 0, 1).float() / 255.0

        mvs_np = np.array([f['motion'] for f in frames])
        mvs_tensor = torch.from_numpy(mvs_np).permute(3, 0, 1, 2).float()

        return iframe_tensor, mvs_tensor