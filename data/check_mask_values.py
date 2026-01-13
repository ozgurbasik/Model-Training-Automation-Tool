"""Check what class values are in validation masks."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SegmentationDataset
from torch.utils.data import DataLoader

# Config
config_path = Path("configs/train_config.yaml")
import yaml
with open(config_path) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]
splits_dir = Path(data_cfg["splits_dir"]).resolve()
val_list = splits_dir / "val.txt"
masks_root = Path(data_cfg["segmentation_masks_root"]).resolve()
dataset_root = Path(data_cfg.get("dataset_root", "")).expanduser()
if not dataset_root.is_absolute():
    dataset_root = (Path(__file__).resolve().parent.parent / dataset_root).resolve()

# Load dataset
val_ds = SegmentationDataset(val_list, masks_root, dataset_root=dataset_root, split_name="val")
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# Check mask values
unique_values = set()
sample_count = 0

for images, masks in val_loader:
    mask_np = masks.numpy()
    unique_values.update(np.unique(mask_np).tolist())
    sample_count += 1
    if sample_count >= 50:
        break

print(f"✅ Checked {sample_count} validation samples")
print(f"📊 Unique class values in masks: {sorted(unique_values)}")
print(f"   Min: {min(unique_values)}, Max: {max(unique_values)}")
print(f"   Total unique values: {len(unique_values)}")

# Load classes
classes_file = masks_root / "classes.txt"
class_names = []
if classes_file.exists():
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

print(f"\n📋 Expected classes ({len(class_names)}): {class_names}")
print(f"⚠️  Values >= {len(class_names)} will be IGNORED in IoU calculation!")

# Check how many pixels would be ignored
ignored_count = 0
total_count = 0
for images, masks in val_loader:
    mask_np = masks.numpy()
    total_count += mask_np.size
    ignored_count += np.sum(mask_np >= len(class_names))

if total_count > 0:
    ignored_pct = (ignored_count / total_count) * 100
    print(f"\n🚨 {ignored_count:,} / {total_count:,} pixels ({ignored_pct:.2f}%) have class index >= {len(class_names)}")
    print(f"   These pixels are IGNORED in IoU calculation!")
