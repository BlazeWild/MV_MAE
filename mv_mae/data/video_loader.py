import av
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class VideoLoader:
    def __init__(self, video_path, gop_size=16):
        """
        Extracts compressed domain data efficiently using a 'Single Pass + Slice' method.
        Ideal for short clips (3-5s) where seeking is unstable due to single I-frame GOPs.
        """
        self.video_path = video_path
        self.gop_size = gop_size
        self.width = 0
        self.height = 0

    def rasterize_mvs(self, mvs, height, width):
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
        try:
            container = av.open(self.video_path, options={'export_mvs': 'true'})
            stream = container.streams.video[0]
            
            try: stream.codec_context.flags2 |= 268435456
            except AttributeError: pass

            self.height = stream.codec_context.height
            self.width = stream.codec_context.width
            
            # Determine total frames to calculate slice anchors
            total_frames = stream.frames
            if total_frames == 0:
                fps = float(stream.average_rate)
                duration_sec = float(stream.duration * stream.time_base) if stream.duration else (container.duration / 1000000.0)
                total_frames = int(duration_sec * fps)
            
            if total_frames <= 0:
                logger.warning(f"Invalid frame count for {self.video_path}.")
                return None, None

            # Calculate 8 mathematical starting indices (anchors) for the chunks
            max_start = max(0, total_frames - self.gop_size)
            if max_start == 0:
                anchor_indices = [0] * num_segments
            else:
                anchor_indices = [int(x) for x in np.linspace(0, max_start, num_segments)]

            frame_buffer = [] 
            frame_idx = 0
            
            # Optimization: Stop decoding entirely once we pass the end of the last required chunk
            read_limit = anchor_indices[-1] + self.gop_size

            # The Single Linear Pass
            for frame in container.decode(stream):
                if frame_idx >= read_limit:
                    break 
                
                mvs = frame.side_data.get('MOTION_VECTORS', [])
                mv_data = self.rasterize_mvs(mvs, self.height, self.width)
                
                # CPU SAVER: Only decode the heavy RGB image if this frame is an anchor
                if frame_idx in anchor_indices:
                    img_data = frame.to_ndarray(format='rgb24').copy()
                else:
                    img_data = None
                    
                frame_buffer.append({'image': img_data, 'motion': mv_data})
                frame_idx += 1
                
            container.close()

            # The Slicer: Build the final tensors
            iframe_tensors = []
            mvs_tensors = []

            for anchor in anchor_indices:
                # Handle edge cases if the video ended slightly earlier than estimated
                if anchor >= len(frame_buffer):
                    anchor = max(0, len(frame_buffer) - self.gop_size)
                    
                chunk = frame_buffer[anchor : anchor + self.gop_size]
                iframe_t, mvs_t = self._pack_and_pad_chunk(chunk)
                
                if iframe_t is not None and mvs_t is not None:
                    iframe_tensors.append(iframe_t)
                    mvs_tensors.append(mvs_t)

            if len(iframe_tensors) != num_segments:
                return None, None

            return torch.stack(iframe_tensors), torch.stack(mvs_tensors)

        except Exception as e:
            logger.error(f"Failed to process {self.video_path}: {e}")
            return None, None

    def _pack_and_pad_chunk(self, chunk):
        actual_len = len(chunk)
        if actual_len == 0:
            return None, None
            
        # Fallback: If the anchor frame failed RGB extraction, generate a black frame
        if chunk[0]['image'] is None:
            chunk[0]['image'] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # Post-padding if the video ends before the 16-frame chunk completes
        if actual_len < self.gop_size:
            padding_needed = self.gop_size - actual_len
            zero_motion = np.zeros_like(chunk[-1]['motion'])
            for _ in range(padding_needed):
                chunk.append({'image': None, 'motion': zero_motion})
                
        iframe_np = chunk[0]['image']
        iframe_tensor = torch.from_numpy(iframe_np).permute(2, 0, 1).float() / 255.0

        mvs_np = np.array([f['motion'] for f in chunk])
        mvs_tensor = torch.from_numpy(mvs_np).permute(3, 0, 1, 2).float()

        return iframe_tensor, mvs_tensor