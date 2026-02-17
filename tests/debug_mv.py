import av
import sys

def test_mv_extraction(video_path):
    print(f"Testing {video_path} with PyAV {av.__version__}")
    
    # Try different option permutations
    configs = [
        {'export_mvs': 'true'},
        {'export_mvs': '1'},
        {'flags2': '+export_mvs'}
    ]

    for conf in configs:
        print(f"\nTrying options={conf}...")
        try:
            container = av.open(video_path, options=conf)
            stream = container.streams.video[0]
            
            # Use codec_context property if available (legacy check)
            try:
                stream.codec_context.export_mvs = True
                print("  Set stream.codec_context.export_mvs = True")
            except:
                pass

            # Inspect Codec Context on first run
            if conf == configs[0]:
                print(f"Codec Context Dir: {[d for d in dir(stream.codec_context) if 'export' in d or 'flags' in d]}")
            
            # ATTEMPT: Manually set AV_CODEC_FLAG2_EXPORT_MVS (1 << 28)
            try:
                # 0x10000000 = 268435456
                stream.codec_context.flags2 |= 268435456
                print("  Manually set flags2 |= 0x10000000")
            except Exception as e:
                print(f"  Could not set flags2: {e}")

            count = 0
            for i, frame in enumerate(container.decode(stream)):
                print(f"  Frame {i} Type: {frame.pict_type}, Side Data Keys: {list(frame.side_data.keys())}")
                vectors = frame.side_data.get('MOTION_VECTORS')
                if vectors is not None:
                    print(f"    Vectors found: {len(vectors)}")
                    if len(vectors) > 0:
                        count += 1
                if i > 5: break
            
            if count > 0:
                print("  SUCCESS: Motion vectors found!")
            else:
                print("  FAILURE: No motion vectors found.")
                
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_mv_extraction("training_fixed.mp4")
