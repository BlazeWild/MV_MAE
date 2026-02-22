import numpy as np

iframes = np.load("tests/tensor_dump/iframes.npy")
mvs = np.load("tests/tensor_dump/mvs.npy")

print(f"Loaded iframes: {iframes.shape}")
print(f"Loaded mvs: {mvs.shape}")

N = iframes.shape[0]

identical_iframes = True
identical_mvs = True

for i in range(1, N):
    if not np.array_equal(iframes[0], iframes[i]):
        identical_iframes = False
        print(f"Iframe 0 and {i} are DIFFERENT")
    else:
        print(f"Iframe 0 and {i} are IDENTICAL")

    if not np.array_equal(mvs[0], mvs[i]):
        identical_mvs = False
        print(f"MVS 0 and {i} are DIFFERENT")
    else:
        print(f"MVS 0 and {i} are IDENTICAL")

if identical_iframes:
    print("\nWARNING: All 8 IFrames are EXACTLY THE SAME.")
else:
    print("\nSUCCESS: IFrames vary across the 8 GOPs.")

if identical_mvs:
    print("WARNING: All 8 Motion Vector grids are EXACTLY THE SAME.")
else:
    print("SUCCESS: Motion Vectors vary across the 8 GOPs.")
