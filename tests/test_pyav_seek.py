import av
import sys

container = av.open("tests/training_fixed_480p.mp4")
stream = container.streams.video[0]

print("Video Duration:", container.duration)
print("Testing seek to arbitrary times...")

timestamps = [0, container.duration // 4, container.duration // 2, container.duration * 3 // 4]

for ts in timestamps:
    try:
        # If we use any_frame=True, PyAV will seek to exact frame but might not decode cleanly.
        container.seek(ts, stream=stream, backward=False, any_frame=False)
        frame = next(container.decode(stream))
        print(f"Seek TS: {ts} -> Decoded frame PTS: {frame.pts}, Keyframe: {frame.key_frame}")
    except Exception as e:
        print(e)
