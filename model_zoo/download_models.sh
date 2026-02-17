#!/bin/bash

# 1. Define input file
INPUT="urls.txt"

# 2. Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found! Ensure you are running this inside the model_zoo folder."
    exit 1
fi

# 3. Read and Download
while read -r folder filename url; do
    # Skip empty lines
    [[ -z "$folder" ]] && continue

    # Create the sub-folder (clip_context/ or video_mae/)
    mkdir -p "$folder"

    echo "Downloading $url..."
    echo "Saving as: $folder/$filename"
    
    # -O forces the rename to your specific .pth name
    wget -q --show-progress -nc -O "$folder/$filename" "$url"

done < "$INPUT"

echo "Done! Your models are saved with the architecture names as .pth files."