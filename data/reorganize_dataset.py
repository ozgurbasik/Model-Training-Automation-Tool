import shutil
from pathlib import Path

def reorganize_dataset(source_dir: Path, target_dir: Path):
    """Reorganize dataset: images and labels into separate folders."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    images_dir = target_dir / "images"
    labels_dir = target_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image and annotation directories
    image_dirs = []
    annotation_dirs = []
    
    for item in source_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.endswith("_Annotations"):
            annotation_dirs.append(item)
        else:
            # Check if corresponding annotation dir exists
            ann_dir = source_dir / (item.name + "_Annotations")
            if ann_dir.exists():
                image_dirs.append(item)
    
    print(f"Found {len(image_dirs)} image directories and {len(annotation_dirs)} annotation directories")
    
    # Copy image directories
    for img_dir in image_dirs:
        print(f"Copying images from {img_dir.name}...")
        target_img_dir = images_dir / img_dir.name
        if target_img_dir.exists():
            print(f"  Warning: {target_img_dir} already exists, skipping...")
            continue
        
        shutil.copytree(img_dir, target_img_dir)
        file_count = len(list(img_dir.glob('*.*')))
        print(f"  [OK] Copied {file_count} files to {target_img_dir}")
    
    # Copy annotation directories
    for ann_dir in annotation_dirs:
        print(f"Copying labels from {ann_dir.name}...")
        # Remove _Annotations suffix for target name
        target_name = ann_dir.name.replace("_Annotations", "")
        target_label_dir = labels_dir / target_name
        if target_label_dir.exists():
            print(f"  Warning: {target_label_dir} already exists, skipping...")
            continue
        
        shutil.copytree(ann_dir, target_label_dir)
        json_count = len(list(ann_dir.glob('*.json')))
        print(f"  [OK] Copied {json_count} JSON files to {target_label_dir}")
    
    print(f"\n[OK] Reorganization complete!")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")


if __name__ == "__main__":
    import sys
    
    source = Path("Dataset")
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    
    target = Path("Dataset_Reorganized")
    if len(sys.argv) > 2:
        target = Path(sys.argv[2])
    
    print(f"Source: {source}")
    print(f"Target: {target}")
    print()
    
    reorganize_dataset(source, target)

