import os
import gdown
import subprocess

# The folder structure you requested
BASE_DIR = "datasets/UAVHuman"
VIDEO_DIR = os.path.join(BASE_DIR, "Action_Videos")

# Dictionary of filenames and their Google Drive IDs extracted from your links
files = {
    "all_rgb.zip.001": "1pgWUmHq9smVwXj46re1E0pfJJwR9NFSk",
    "all_rgb.zip.002": "1pzte2yP2OPni9gtlrbJfebnWLhn70xBA",
    "all_rgb.zip.003": "1dJ5sj1YNQxHwSV22A9GTNQfs68YadIq3",
    "all_rgb.zip.004": "1g17nz8i0eE6gh7R-eEL3PLJtRfQg5sHC",
    "all_rgb.zip.005": "1yQHwIzbeuX2rb5uM9Z-ke9Zm1Mk19b54",
    "all_rgb.zip.006": "1R7f7arweOG8VTlpXmjJgeQz9zl7ZXe2Q",
    "all_rgb.zip.007": "1TFiLSObJKBrHchdlE6hRsRbuXlU817YF",
    "all_rgb.zip.008": "1JZKN1LxF7HhUMkIj23RJc2uRQQJLwpmy",
    "all_rgb.zip.009": "1NZpfa-002AZ3F53k5QdNj7jqawXn7Ojy",
    "all_rgb.zip.010": "1URIzR337IyfdbGaa03LZh9HN6PB07ePm",
}

def setup_structure():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    print(f"✅ Created directory: {VIDEO_DIR}")

def download_files():
    for filename, file_id in files.items():
        output_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(output_path):
            print(f"⬇️ Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
        else:
            print(f"⏩ {filename} already exists, skipping download.")

def extract_files():
    print("🔗 Concatenating split zip files...")
    # Navigate to the directory to run shell commands
    os.chdir(BASE_DIR)
    
    # Combine parts: all_rgb.zip.001 + .002 ... -> all_rgb_full.zip
    # Using shell=True to handle the wildcard *
    subprocess.run("cat all_rgb.zip.* > all_rgb_full.zip", shell=True, check=True)
    
    print("📦 Extracting videos (this will take a long time)...")
    # Extract into the Action_Videos folder
    subprocess.run(["unzip", "-q", "all_rgb_full.zip", "-d", "Action_Videos/"], check=True)
    
    # Cleanup large zip files to save space
    print("🧹 Cleaning up zip archives to save disk space...")
    subprocess.run("rm all_rgb.zip.* all_rgb_full.zip", shell=True, check=True)

if __name__ == "__main__":
    setup_structure()
    download_files()
    extract_files()
    print(f"✨ Done! Your videos are in {VIDEO_DIR}")