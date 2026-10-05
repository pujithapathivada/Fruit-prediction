"""Dataset splitting pipeline for AgriFreshNET Processed_Data.

Splits 24 agricultural produce classes into train (80%), validation (10%),
and test (10%) sets with a fixed random seed (42).

Preserves original folder names and does not modify the source dataset.
Uses Python standard libraries: pathlib, shutil, random, and sys.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Source and destination defaults
DEFAULT_SOURCE_PATH = Path(r"C:\Users\pujit\Downloads\Processed Data\Processed Data")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST_PATH = PROJECT_ROOT / "dataset"

# Supported image file formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def get_class_folders(source_dir: Path) -> List[Path]:
    """Retrieves all valid class subfolders from the source dataset directory."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dir}")

    folders = sorted(
        [
            p for p in source_dir.iterdir()
            if p.is_dir() and not p.name.startswith((".", "__"))
        ],
        key=lambda p: p.name.lower(),
    )
    return folders


def copy_split_files(
    file_list: List[Path],
    destination_folder: Path,
) -> Tuple[int, int]:
    """Copies image files to destination folder safely without duplicating existing files.

    Returns:
        (copied_count, skipped_existing_count)
    """
    destination_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for src_file in file_list:
        dst_file = destination_folder / src_file.name
        # If destination file already exists with same size, skip copying
        if dst_file.exists():
            if dst_file.stat().st_size == src_file.stat().st_size:
                skipped += 1
                continue

        shutil.copy2(src_file, dst_file)
        copied += 1

    return copied, skipped


def split_dataset(
    source_dir: Path = DEFAULT_SOURCE_PATH,
    dest_dir: Path = DEFAULT_DEST_PATH,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, int]]:
    """Splits each class into 80% train, 10% validation, and 10% test partitions.

    Args:
        source_dir: Directory containing original 24 class folders.
        dest_dir: Directory where train/, validation/, and test/ will be created.
        seed: Random seed for reproducible splitting.

    Returns:
        Dictionary containing partition statistics per class.
    """
    print("=" * 80)
    print(" AgriFreshNET Dataset Split Pipeline (Train 80% | Val 10% | Test 10%)")
    print("=" * 80)
    print(f"Source Directory      : {source_dir}")
    print(f"Destination Directory : {dest_dir}")
    print(f"Random Seed           : {seed}\n")

    train_base = dest_dir / "train"
    val_base = dest_dir / "validation"
    test_base = dest_dir / "test"

    for base in (train_base, val_base, test_base):
        base.mkdir(parents=True, exist_ok=True)

    class_folders = get_class_folders(source_dir)
    if not class_folders:
        print("[ERROR] No class folders found in source directory.")
        return {}

    print(f"[i] Found {len(class_folders)} class folders. Beginning reproducible split...\n")

    header = f"{'No.':<4} | {'Class Name':<32} | {'Train':>7} | {'Val':>6} | {'Test':>6} | {'Total':>7}"
    print(header)
    print("-" * len(header))

    summary_stats = {
        "train": 0,
        "validation": 0,
        "test": 0,
        "total": 0,
        "skipped": 0,
    }

    per_class_results = {}

    # Initialize reproducible RNG
    rng = random.Random(seed)

    for idx, class_folder in enumerate(class_folders, start=1):
        # 1. Deterministically sort images first to ensure platform-independent order
        images = sorted(
            [
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda f: f.name,
        )

        n_images = len(images)
        if n_images == 0:
            print(f"{idx:<4} | {class_folder.name:<32} | {'0':>7} | {'0':>6} | {'0':>6} | {'0':>7}")
            continue

        # 2. Shuffle using seeded RNG
        shuffled = list(images)
        rng.shuffle(shuffled)

        # 3. Calculate partition sizes
        n_train = int(n_images * TRAIN_RATIO)
        n_val = int(n_images * VAL_RATIO)
        n_test = n_images - n_train - n_val

        train_files = shuffled[:n_train]
        val_files = shuffled[n_train : n_train + n_val]
        test_files = shuffled[n_train + n_val :]

        # 4. Copy to train, validation, and test (preserving exact folder name)
        c_tr, s_tr = copy_split_files(train_files, train_base / class_folder.name)
        c_va, s_va = copy_split_files(val_files, val_base / class_folder.name)
        c_te, s_te = copy_split_files(test_files, test_base / class_folder.name)

        summary_stats["train"] += len(train_files)
        summary_stats["validation"] += len(val_files)
        summary_stats["test"] += len(test_files)
        summary_stats["total"] += n_images
        summary_stats["skipped"] += (s_tr + s_va + s_te)

        per_class_results[class_folder.name] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
            "total": n_images,
        }

        print(
            f"{idx:<4} | {class_folder.name:<32} | {len(train_files):>7} | {len(val_files):>6} | {len(test_files):>6} | {n_images:>7}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<39} | {summary_stats['train']:>7,} | {summary_stats['validation']:>6,} | {summary_stats['test']:>6,} | {summary_stats['total']:>7,}"
    )
    print("=" * 80)

    print("\nDataset Split Summary:")
    print(f"  • Train Set        (80%) : {summary_stats['train']:,} images ({train_base})")
    print(f"  • Validation Set   (10%) : {summary_stats['validation']:,} images ({val_base})")
    print(f"  • Test Set         (10%) : {summary_stats['test']:,} images ({test_base})")
    print(f"  • Grand Total            : {summary_stats['total']:,} images across {len(class_folders)} classes")

    if summary_stats["skipped"] > 0:
        print(f"  • Note: {summary_stats['skipped']:,} images already existed in split folders and were safely preserved.")

    print("\n[OK] Dataset split completed successfully without modifying source dataset.")
    return per_class_results


def main():
    parser = argparse.ArgumentParser(description="Split AgriFreshNET dataset into train, val, and test.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Path to source AgriFreshNET dataset.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST_PATH,
        help="Path to target destination dataset folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic split.",
    )
    args = parser.parse_args()

    split_dataset(source_dir=args.source, dest_dir=args.dest, seed=args.seed)


if __name__ == "__main__":
    main()
