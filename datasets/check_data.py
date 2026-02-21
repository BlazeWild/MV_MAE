import os
import collections

# ── CONFIG ────────────────────────────────────────────────────────────────────
UAV_ROOT = os.path.join(os.path.dirname(__file__), "UAVHuman")
# ──────────────────────────────────────────────────────────────────────────────


def scan_file_types(root: str):
    """Walk the UAVHuman tree and count every file extension."""
    ext_counter = collections.Counter()
    all_files = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower() or "(no extension)"
            ext_counter[ext] += 1
            all_files.append(os.path.join(dirpath, fname))

    return ext_counter, all_files


def show_non_video_files(all_files: list):
    """Print content of every non-AVI file found."""
    VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    non_video = [f for f in all_files if os.path.splitext(f)[1].lower() not in VIDEO_EXTS]

    if not non_video:
        print("  ✗ No non-video files found inside UAVHuman/")
        return

    for fpath in non_video:
        rel = os.path.relpath(fpath, UAV_ROOT)
        ext = os.path.splitext(fpath)[1].lower()
        size = os.path.getsize(fpath)
        print(f"\n{'─'*70}")
        print(f"  FILE : {rel}  ({size:,} bytes)")
        print(f"{'─'*70}")

        # Try to read as text
        try:
            with open(fpath, "r", errors="replace") as f:
                content = f.read(4000)          # first 4 KB
            if len(content) == 4000:
                content += "\n  … [truncated] …"
            print(content)
        except Exception as e:
            print(f"  [Could not read file: {e}]")


def show_avi_sample(all_files: list, n: int = 10):
    """Show a few AVI filenames so we can understand the naming convention."""
    avis = [f for f in all_files if f.lower().endswith(".avi")]
    print(f"\n  Sample AVI filenames ({min(n, len(avis))} of {len(avis):,}):")
    for f in avis[:n]:
        print(f"    {os.path.basename(f)}")


def main():
    print("=" * 70)
    print(f"  Scanning: {UAV_ROOT}")
    print("=" * 70)

    if not os.path.isdir(UAV_ROOT):
        print(f"\n  ERROR: Directory not found → {UAV_ROOT}")
        print("  Make sure the UAVHuman dataset is downloaded first.")
        return

    ext_counter, all_files = scan_file_types(UAV_ROOT)

    # ── 1. File-type summary ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  FILE TYPE BREAKDOWN")
    print(f"{'─'*70}")
    for ext, count in ext_counter.most_common():
        bar = "█" * min(count // 100, 40)
        print(f"  {ext:>20}  {count:>6,}  {bar}")
    print(f"  {'TOTAL':>20}  {sum(ext_counter.values()):>6,}")

    # ── 2. Non-video files (read + display) ──────────────────────────────────
    print(f"\n{'─'*70}")
    print("  NON-VIDEO FILES (contents shown below)")
    print(f"{'─'*70}")
    show_non_video_files(all_files)

    # ── 3. AVI filename sample ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  AVI FILENAME SAMPLE  (naming convention)")
    print(f"{'─'*70}")
    show_avi_sample(all_files)

    print(f"\n{'='*70}")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()