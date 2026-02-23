"""
Upload UAVHuman_480p_mp4 dataset to Hugging Face using upload_large_folder.

Uses num_workers=1 to avoid hitting API rate limits (429 errors).
Automatically resumes from .cache/huggingface metadata if a previous run
was interrupted.
"""

import os
import time
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not set in environment or .env file")

api = HfApi(token=token)

print("Starting upload (num_workers=1 to avoid rate limits)...")
print("This will resume from any previously cached metadata.")
print("Press Ctrl+C to interrupt — you can resume later by re-running.\n")

api.upload_large_folder(
    repo_id="Blazewild/Blaze_480p_actionrecog",
    repo_type="dataset",
    folder_path="/home/blaze/MV_MAE/datasets/UAVHuman_480p_mp4",
    num_workers=1,
)

print("\nUpload complete!")
