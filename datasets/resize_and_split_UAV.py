import os
import re
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = "datasets/UAVHuman/Action_Videos/all_rgb"
DATA_DIR = "datasets/UAVHuman_240p_mp4"
VIDEO_DIR = os.path.join(DATA_DIR, "Action_Videos")

FFMPEG_CMD = "ffmpeg"

# GPU encode (h264_nvenc) saturates the NVENC engine quickly;
MAX_WORKERS = 24

# -------------------------------------------------------------------------
# NEW SPLIT LOGIC: 
# Keep Test untouched for official comparison. Split original Train into Train/Val.
# -------------------------------------------------------------------------

# The 89 Official Cross-Subject-v1 Training IDs (from the CVPR paper)
OFFICIAL_TRAIN_SUBJECTS = {
    0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26,
    27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70,
    71, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100,
    102, 103, 105, 106, 110, 111, 112, 114, 115, 116, 117, 118
}

# Take 9 subjects (~10%) from the official train set for Validation
VAL_SUBJECTS = {106, 110, 111, 112, 114, 115, 116, 117, 118}

# The remaining 80 subjects become your new Training set
TRAIN_SUBJECTS = OFFICIAL_TRAIN_SUBJECTS - VAL_SUBJECTS

# (The 30 remaining subjects not listed above will automatically become the Test set)

# -------------------------------------------------------------------------
# GPU availability check
# -------------------------------------------------------------------------
def _has_nvenc() -> bool:
    """Return True if ffmpeg can use h264_nvenc on this machine."""
    try:
        result = subprocess.run(
            [FFMPEG_CMD, '-hide_banner', '-encoders'],
            capture_output=True, text=True, check=True
        )
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False

USE_GPU = _has_nvenc()

# -------------------------------------------------------------------------
# Step 1: Resize and Convert (.avi -> 240p .mp4)
# -------------------------------------------------------------------------
def resize_video(vid_path):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    filename = os.path.basename(vid_path)

    # Change extension to .mp4
    out_name = filename.rsplit('.', 1)[0] + '.mp4'
    out_path = os.path.join(VIDEO_DIR, out_name)

    if os.path.exists(out_path):
        return True

    if USE_GPU:
        cmd = [
            FFMPEG_CMD, '-y', '-hide_banner', '-loglevel', 'error',
            '-i', vid_path,
            '-vf', 'scale=-2:240',
            '-threads', '2',        
            '-filter_threads', '2', 
            '-c:v', 'h264_nvenc',
            '-g', '16', '-keyint_min', '16',
            '-preset', 'p1',    
            '-rc', 'vbr',
            '-cq', '23',        
            out_path
        ]
    else:
        print("⚠️  No NVENC found — falling back to CPU libx264 (slow).")
        cmd = [
            FFMPEG_CMD, '-y', '-hide_banner', '-loglevel', 'error',
            '-i', vid_path,
            '-vf', 'scale=-2:240',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            out_path
        ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def run_conversion():
    mode = "GPU (h264_nvenc)" if USE_GPU else "CPU (libx264)"
    print(f"🎥 STEP 1: Scanning for raw .avi videos in {INPUT_DIR}...")
    print(f"⚙️  Encoder: {mode}  |  Workers: {MAX_WORKERS}")
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.avi"))

    if not video_files:
        print("❌ No .avi videos found! Check your input directory.")
        return False

    print(f"⚡ Resizing {len(video_files)} videos to 240p .mp4 using {MAX_WORKERS} worker(s)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(resize_video, video_files), total=len(video_files)))

    success_count = sum(1 for r in results if r)
    print(f"✅ Conversion complete! {success_count}/{len(video_files)} videos processed.")
    return True

# -------------------------------------------------------------------------
# Step 2: Build the CSv1 Splits (Updated for Train, Val, Test)
# -------------------------------------------------------------------------
def build_splits():
    print("\n🗂️ STEP 2: Building Train, Val, and official Test splits...")
    train_file = os.path.join(DATA_DIR, "train_split.txt")
    val_file = os.path.join(DATA_DIR, "val_split.txt")
    test_file = os.path.join(DATA_DIR, "test_split.txt")

    train_count, val_count, test_count = 0, 0, 0

    with open(train_file, 'w') as f_train, open(val_file, 'w') as f_val, open(test_file, 'w') as f_test:
        for filename in os.listdir(VIDEO_DIR):
            if not filename.endswith(".mp4"):
                continue

            person_match = re.search(r'P(\d{3})', filename)
            action_match = re.search(r'A(\d{3})', filename)

            if person_match and action_match:
                person_id = int(person_match.group(1))
                action_id = int(action_match.group(1))

                line = f"{filename} {action_id}\n"

                # Sort into the correct file
                if person_id in TRAIN_SUBJECTS:
                    f_train.write(line)
                    train_count += 1
                elif person_id in VAL_SUBJECTS:
                    f_val.write(line)
                    val_count += 1
                else:
                    # Anything not in Train or Val is the official untouched Test set
                    f_test.write(line)
                    test_count += 1

    print(f"✅ Splits created successfully in {DATA_DIR}!")
    print(f"🎓 Training samples:   {train_count}")
    print(f"🧪 Validation samples: {val_count}")
    print(f"📋 Test samples:       {test_count}")

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    if run_conversion():
        build_splits()
        print("\n🚀 DATA PREPARATION FULLY COMPLETE. Ready for training.")