"""Verification script for dataset split integrity.

Verifies that dataset/train, dataset/validation, and dataset/test:
1. Exist and contain exactly the same 24 class folders.
2. Have exact expected image counts: Train = 472, Validation = 59, Test = 59 per class.
3. Prints "DATASET SPLIT VERIFIED SUCCESSFULLY" only when all checks pass.

Safety: Read-only inspection. Does not modify any files and does not train any model.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Dataset paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Expected targets per class
EXPECTED_CLASSES_COUNT = 24
EXPECTED_TRAIN_COUNT = 472
EXPECTED_VAL_COUNT = 59
EXPECTED_TEST_COUNT = 59


def count_images_in_folder(folder_path: Path) -> int:
    """Counts valid image files in a directory."""
    if not folder_path.exists() or not folder_path.is_dir():
        return 0
    return sum(
        1 for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_class_folders(split_dir: Path) -> List[str]:
    """Retrieves class folder names inside a split directory."""
    if not split_dir.exists():
        return []
    return sorted(
        [
            p.name for p in split_dir.iterdir()
            if p.is_dir() and not p.name.startswith((".", "__"))
        ],
        key=lambda s: s.lower(),
    )


def verify_dataset_split() -> bool:
    """Performs comprehensive verification of the dataset split."""
    print("=" * 75)
    print(" Dataset Split Verification")
    print("=" * 75)
    print(f"Dataset Root: {DATASET_DIR.resolve()}\n")

    # 1. Check directories existence
    splits = {
        "Train": TRAIN_DIR,
        "Validation": VAL_DIR,
        "Test": TEST_DIR,
    }

    all_splits_exist = True
    for name, path in splits.items():
        if not path.exists() or not path.is_dir():
            print(f"[ERROR] {name} directory missing at: {path}")
            all_splits_exist = False

    if not all_splits_exist:
        return False

    # 2. Retrieve classes in each split
    train_classes = get_class_folders(TRAIN_DIR)
    val_classes = get_class_folders(VAL_DIR)
    test_classes = get_class_folders(TEST_DIR)

    # 3. Print number of classes in each split
    print(f"Class count in Train      : {len(train_classes)}")
    print(f"Class count in Validation : {len(val_classes)}")
    print(f"Class count in Test       : {len(test_classes)}\n")

    all_passed = True

    # Check 24 class count
    if len(train_classes) != EXPECTED_CLASSES_COUNT:
        print(f"[FAILED] Train has {len(train_classes)} classes, expected {EXPECTED_CLASSES_COUNT}")
        all_passed = False

    if len(val_classes) != EXPECTED_CLASSES_COUNT:
        print(f"[FAILED] Validation has {len(val_classes)} classes, expected {EXPECTED_CLASSES_COUNT}")
        all_passed = False

    if len(test_classes) != EXPECTED_CLASSES_COUNT:
        print(f"[FAILED] Test has {len(test_classes)} classes, expected {EXPECTED_CLASSES_COUNT}")
        all_passed = False

    # 4. Verify class names are identical across all three splits
    set_train = set(train_classes)
    set_val = set(val_classes)
    set_test = set(test_classes)

    if set_train != set_val or set_train != set_test:
        print("[FAILED] Class names are not identical across all three splits!")
        diff_val = set_train.symmetric_difference(set_val)
        diff_test = set_train.symmetric_difference(set_test)
        if diff_val:
            print(f"  Differences between Train and Validation: {diff_val}")
        if diff_test:
            print(f"  Differences between Train and Test: {diff_test}")
        all_passed = False
    else:
        print("[PASS] All three splits contain exactly identical class names.")

    # 5. Print table and verify counts for every class
    header = f"{'No.':<4} | {'Class Name':<32} | {'Train':>6} | {'Val':>5} | {'Test':>5} | {'Status':<8}"
    print("\n" + header)
    print("-" * len(header))

    total_train = 0
    total_val = 0
    total_test = 0

    for idx, class_name in enumerate(train_classes, start=1):
        c_train = count_images_in_folder(TRAIN_DIR / class_name)
        c_val = count_images_in_folder(VAL_DIR / class_name)
        c_test = count_images_in_folder(TEST_DIR / class_name)

        total_train += c_train
        total_val += c_val
        total_test += c_test

        is_class_valid = (
            c_train == EXPECTED_TRAIN_COUNT
            and c_val == EXPECTED_VAL_COUNT
            and c_test == EXPECTED_TEST_COUNT
        )

        status_str = "[OK]" if is_class_valid else "[FAIL]"
        if not is_class_valid:
            all_passed = False

        print(f"{idx:<4} | {class_name:<32} | {c_train:>6} | {c_val:>5} | {c_test:>5} | {status_str:<8}")

    print("-" * len(header))
    print(f"{'TOTAL':<39} | {total_train:>6,} | {total_val:>5,} | {total_test:>5,} |")
    print("=" * 75)

    # 6. Check overall totals
    expected_total_train = EXPECTED_CLASSES_COUNT * EXPECTED_TRAIN_COUNT
    expected_total_val = EXPECTED_CLASSES_COUNT * EXPECTED_VAL_COUNT
    expected_total_test = EXPECTED_CLASSES_COUNT * EXPECTED_TEST_COUNT

    print(f"\nTarget Verification:")
    print(f"• Train Total      : {total_train:,} / {expected_total_train:,} (Expected 472/class)")
    print(f"• Validation Total : {total_val:,} / {expected_total_val:,} (Expected 59/class)")
    print(f"• Test Total       : {total_test:,} / {expected_total_test:,} (Expected 59/class)")

    if (
        total_train != expected_total_train
        or total_val != expected_total_val
        or total_test != expected_total_test
    ):
        all_passed = False

    # 7. Print final verdict
    print("\n" + "=" * 75)
    if all_passed:
        print("DATASET SPLIT VERIFIED SUCCESSFULLY")
        print("=" * 75)
        return True
    else:
        print("[FAILED] Dataset split verification encountered discrepancies.")
        print("=" * 75)
        return False


def main():
    success = verify_dataset_split()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
