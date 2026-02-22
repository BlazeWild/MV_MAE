import av
import sys

container = av.open("tests/training_fixed_480p.mp4", options={'export_mvs': 'true'})
stream = container.streams.video[0]

count = 0
iframe_count = 0

for frame in container.decode(stream):
    if frame.key_frame:
        print(f"I-Frame at PTS: {frame.pts}, Time: {frame.time}")
        iframe_count += 1
    count += 1

print(f"Total Frames Decoded: {count}")
print(f"Total I-Frames: {iframe_count}")
container.close()
