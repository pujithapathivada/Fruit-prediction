"""Dataset verification script for AgriFreshNET Processed_Data.

Uses standard Python and pathlib to inspect the dataset folder, identify class subfolders,
and count image samples without modifying, moving, copying, or deleting any files.
"""

import sys
from pathlib import Path

# Dataset location provided for AgriFreshNET Processed Data
DATASET_PATH = Path(r"C:\Users\pujit\Downloads\Processed Data\Processed Data")

# Valid image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def check_dataset(dataset_dir: Path) -> None:
    """Inspects the dataset directory, detects classes, and counts images."""
    print("=" * 65)
    print(" AgriFreshNET Dataset Verification")
    print("=" * 65)
    print(f"Target Directory: {dataset_dir}\n")

    # 1. Check whether the folder exists
    if not dataset_dir.exists():
        print(f"[ERROR] The specified folder does not exist:\n  {dataset_dir}")
        return

    if not dataset_dir.is_dir():
        print(f"[ERROR] The specified path is not a directory:\n  {dataset_dir}")
        return

    # 2. Find all class folders inside it
    class_folders = sorted(
        [
            p for p in dataset_dir.iterdir()
            if p.is_dir() and not p.name.startswith((".", "__"))
        ],
        key=lambda p: p.name.lower(),
    )

    if not class_folders:
        print("[WARNING] No class subfolders found inside the directory.")
        return

    # 3. Count image files in every class folder & print details
    print(f"{'No.':<4} | {'Class Name':<32} | {'Image Count':>12}")
    print("-" * 55)

    total_images = 0

    for idx, folder in enumerate(class_folders, start=1):
        image_count = sum(
            1 for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        total_images += image_count
        print(f"{idx:<4} | {folder.name:<32} | {image_count:>12,}")

    print("-" * 55)
    # 4. Print total classes and total images
    print(f"Total Number of Classes : {len(class_folders)}")
    print(f"Total Number of Images  : {total_images:,}")
    print("=" * 65)


def main():
    # Ensure UTF-8 output encoding on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    check_dataset(DATASET_PATH)


if __name__ == "__main__":
    main()
