from pathlib import Path
import random
import shutil

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# -----------------------------
# Create destination folders
# -----------------------------
for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Process every class
# -----------------------------
class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

print(f"Found {len(class_dirs)} classes.\n")

total_train = 0
total_val = 0
total_test = 0

for class_dir in class_dirs:

    images = [
        file for file in class_dir.iterdir()
        if file.is_file() and file.suffix in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Create class folders
    train_class_dir = TRAIN_DIR / class_dir.name
    val_class_dir = VAL_DIR / class_dir.name
    test_class_dir = TEST_DIR / class_dir.name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for image in train_images:
        shutil.copy2(image, train_class_dir / image.name)

    for image in val_images:
        shutil.copy2(image, val_class_dir / image.name)

    for image in test_images:
        shutil.copy2(image, test_class_dir / image.name)

    total_train += train_count
    total_val += val_count
    total_test += test_count

    print(
        f"{class_dir.name}: "
        f"{train_count} train | "
        f"{val_count} val | "
        f"{test_count} test | "
        f"{total} total"
    )


# -----------------------------
# Final summary
# -----------------------------
print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETE")
print("=" * 70)

print(f"Train images : {total_train}")
print(f"Validation   : {total_val}")
print(f"Test images  : {total_test}")
print(f"Total images : {total_train + total_val + total_test}")

print("\nSplit ratio:")
print(f"Train: {total_train / (total_train + total_val + total_test):.2%}")
print(f"Val  : {total_val / (total_train + total_val + total_test):.2%}")
print(f"Test : {total_test / (total_train + total_val + total_test):.2%}")

print("\nDone.")