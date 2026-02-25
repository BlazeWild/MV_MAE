import os
import gdown
import zipfile
import subprocess
from pathlib import Path

# The folder structure
BASE_DIR = "datasets/UAVHuman"
VIDEO_DIR = os.path.join(BASE_DIR, "Action_Videos")

# Dictionary of filenames and their Google Drive IDs
files = {
    "all_rgb.zip.001": "1pgWUmHq9smVwXj46re1E0pfJJwR9NFSk",
    # "all_rgb.zip.002": "1pzte2yP2OPni9gtlrbJfebnWLhn70xBA",
    # "all_rgb.zip.003": "1dJ5sj1YNQxHwSV22A9GTNQfs68YadIq3",
    # "all_rgb.zip.004": "1g17nz8i0eE6gh7R-eEL3PLJtRfQg5sHC",
    # "all_rgb.zip.005": "1yQHwIzbeuX2rb5uM9Z-ke9Zm1Mk19b54",
    # "all_rgb.zip.006": "1R7f7arweOG8VTlpXmjJgeQz9zl7ZXe2Q",
    # "all_rgb.zip.007": "1TFiLSObJKBrHchdlE6hRsRbuXlU817YF",
    # "all_rgb.zip.008": "1JZKN1LxF7HhUMkIj23RJc2uRQQJLwpmy",
    # "all_rgb.zip.009": "1NZpfa-002AZ3F53k5QdNj7jqawXn7Ojy",
    # "all_rgb.zip.010": "1URIzR337IyfdbGaa03LZh9HN6PB07ePm",
}

class SplitZipReader:
    """A custom file-like object that reads across multiple split files in memory."""
    def __init__(self, file_list):
        self.files = sorted(file_list)
        self.handles = [open(f, 'rb') for f in self.files]
        self.sizes = [os.path.getsize(f) for f in self.files]
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total_size = sum(self.sizes)
        self.pos = 0

    def read(self, size=-1):
        if size < 0:
            size = self.total_size - self.pos
        data = bytearray()
        while size > 0 and self.pos < self.total_size:
            idx = next(i for i, off in reversed(list(enumerate(self.offsets))) if self.pos >= off)
            file_pos = self.pos - self.offsets[idx]
            
            self.handles[idx].seek(file_pos)
            chunk_size = min(size, self.sizes[idx] - file_pos)
            chunk = self.handles[idx].read(chunk_size)
            
            if not chunk: break
                
            data.extend(chunk)
            self.pos += len(chunk)
            size -= len(chunk)
        return bytes(data)

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET: self.pos = offset
        elif whence == os.SEEK_CUR: self.pos += offset
        elif whence == os.SEEK_END: self.pos = self.total_size + offset
        self.pos = max(0, min(self.pos, self.total_size))
        return self.pos

    def tell(self): return self.pos
    def close(self): [h.close() for h in self.handles]

    # Required for Python 3.10+
    def seekable(self): return True
    def readable(self): return True
    def writable(self): return False

def setup_structure():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    print(f"✅ Directory ready: {VIDEO_DIR}")

def download_files():
    print("\n--- Starting Downloads ---")
    for filename, file_id in files.items():
        output_path = os.path.join(BASE_DIR, filename)
        
        # Check if the file is already fully downloaded
        if os.path.exists(output_path):
            # Check if file size is > 0 to avoid skipping corrupted/empty downloads
            if os.path.getsize(output_path) > 1024:
                print(f"⏩ {filename} already exists, skipping download.")
                continue
            else:
                print(f"⚠️ {filename} exists but seems empty/corrupted. Redownloading...")
                
        print(f"⬇️ Downloading {filename}...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

def extract_files():
    print("\n--- Starting Extraction ---")
    base_dir_path = Path(BASE_DIR)
    split_files = sorted(base_dir_path.glob("all_rgb.zip.*"))
    
    if not split_files:
        print("❌ No split zip files found. Did the download fail?")
        return

    print(f"🚀 Found {len(split_files)} split files. Extracting the available parts using 7z...")
    
    # Use 7z to extract as it can handle incomplete split archives
    import subprocess
    first_part = split_files[0]
    try:
        print(f"⏳ Extracting from {first_part.name}...")
        # '-y' assumes Yes to all queries (e.g., overwrite), '-bd' disables percentage indicator
        # '-bso0' could disable stdout, but let's keep it visible so user sees it extracting
        subprocess.run(["7z", "x", "-y", str(first_part), f"-o{VIDEO_DIR}"])
        print("\n\n✅ Extraction process finished for the available subset!")
        
        print("🧹 Keeping downloaded parts to avoid redownloading later.")
            
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")

if __name__ == "__main__":
    setup_structure()
    download_files()
    extract_files()
    print(f"\n✨ Done! Your videos are fully extracted in {VIDEO_DIR}")