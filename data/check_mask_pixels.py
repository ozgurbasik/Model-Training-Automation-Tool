"""Check mask pixel values to identify invalid indices."""

from pathlib import Path
from PIL import Image
import numpy as np

masks_dir = Path("splits_seg/val/labels")
mask_files = sorted(masks_dir.glob("*.png"))[:20]

print(f"Checking {len(mask_files)} mask files...\n")

all_values = set()
for mask_file in mask_files:
    img = Image.open(mask_file)
    mask_array = np.array(img)
    unique = np.unique(mask_array)
    all_values.update(unique.tolist())
    print(f"{mask_file.name}: unique values = {sorted(unique.tolist())}")

print(f"\n{'='*60}")
print(f"📊 SUMMARY:")
print(f"{'='*60}")
print(f"All unique pixel values found: {sorted(all_values)}")
print(f"Min: {min(all_values)}, Max: {max(all_values)}")
print(f"Total unique values: {len(all_values)}")

# Check classes.txt
classes_file = masks_dir / "classes.txt"
if classes_file.exists():
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"\n📋 Expected classes: {len(classes)} classes")
    print(f"   Classes: {classes}")
    print(f"\n⚠️  Values >= {len(classes)} will cause metric inflation!")
    
    invalid = [v for v in all_values if v >= len(classes)]
    if invalid:
        print(f"🚨 FOUND INVALID CLASS INDICES: {sorted(invalid)}")
    else:
        print(f"✅ All pixel values are valid class indices")
