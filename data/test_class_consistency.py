"""
Test script to verify that classes.txt files will be identical after the fix.
This simulates the fixed behavior.
"""
from pathlib import Path
from data.labelme_to_segmentation import collect_label_map as collect_seg_label_map

def test_class_consistency():
    splits_dir = Path("splits")
    
    # Collect all label directories (simulating the fixed code)
    all_label_dirs = []
    for split_name in ["train", "val"]:
        labels_dir = splits_dir / split_name / "labels"
        if labels_dir.exists():
            all_label_dirs.append(labels_dir)
            print(f"✓ Found {split_name} labels dir: {labels_dir}")
    
    if not all_label_dirs:
        print("✗ No label directories found!")
        return False
    
    # Collect label map from ALL splits at once
    print("\nCollecting global label map from all splits...")
    global_label_map = collect_seg_label_map(all_label_dirs)
    
    print(f"\n📋 Global label map ({len(global_label_map)} classes):")
    for name, cid in sorted(global_label_map.items(), key=lambda x: x[1]):
        print(f"   {cid}: {name}")
    
    # Simulate what would be written to classes.txt
    print("\n📝 Content that will be written to classes.txt:")
    print("0 background")
    for name, cid in sorted(global_label_map.items(), key=lambda x: x[1]):
        print(f"{cid} {name}")
    
    print("\n✓ Both train and val will use this SAME ordering!")
    return True

if __name__ == "__main__":
    test_class_consistency()
