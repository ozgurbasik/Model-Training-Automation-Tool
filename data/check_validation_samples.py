"""Compare validation samples used in training vs debug."""

from pathlib import Path

val_list = Path("splits_seg/val.txt")

if val_list.exists():
    with open(val_list) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Found {len(lines)} validation samples in {val_list}")
    print(f"\nFirst 10 samples:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i+1}. {line}")
    print(f"\nLast 5 samples:")
    for i, line in enumerate(lines[-5:]):
        print(f"  {len(lines)-4+i}. {line}")
else:
    print(f"❌ {val_list} not found!")
    
# Also check train list
train_list = Path("splits_seg/train.txt")
if train_list.exists():
    with open(train_list) as f:
        train_lines = [line.strip() for line in f if line.strip()]
    print(f"\n✅ Found {len(train_lines)} training samples in {train_list}")
else:
    print(f"\n❌ {train_list} not found!")
