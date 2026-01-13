import shutil
from pathlib import Path
from collections import defaultdict

def merge_dataset_folders(source_dir: Path, target_images_dir: Path, target_labels_dir: Path):
    """Merge all images and labels from subdirectories into single folders."""
    source_dir = Path(source_dir)
    target_images_dir = Path(target_images_dir)
    target_labels_dir = Path(target_labels_dir)
    
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    images_source = source_dir / "images"
    labels_source = source_dir / "labels"
    
    if not images_source.exists():
        print(f"Error: {images_source} not found!")
        return
    
    if not labels_source.exists():
        print(f"Error: {labels_source} not found!")
        return
    
    # Track file names to handle duplicates
    image_name_counts = defaultdict(int)
    label_name_counts = defaultdict(int)
    
    # Process images
    print("Processing images...")
    for subdir in sorted(images_source.iterdir()):
        if not subdir.is_dir():
            continue
        
        print(f"  Processing {subdir.name}...")
        image_files = list(subdir.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        for img_file in image_files:
            original_name = img_file.name
            base_name = img_file.stem
            extension = img_file.suffix
            
            # Always add prefix (including H1)
            new_name = f"{subdir.name}_{original_name}"
            
            image_name_counts[original_name] += 1
            
            target_path = target_images_dir / new_name
            
            # If target already exists with different content, add counter
            if target_path.exists() and target_path.stat().st_size != img_file.stat().st_size:
                counter = 1
                while target_path.exists():
                    new_name = f"{subdir.name}_{base_name}_{counter}{extension}"
                    target_path = target_images_dir / new_name
                    counter += 1
            
            shutil.copy2(img_file, target_path)
            print(f"    Copied: {img_file.name} -> {new_name}")
    
    print(f"\n  Total images copied: {sum(image_name_counts.values())}")
    
    # Process labels
    print("\nProcessing labels...")
    for subdir in sorted(labels_source.iterdir()):
        if not subdir.is_dir():
            continue
        
        print(f"  Processing {subdir.name}...")
        json_files = list(subdir.glob("*.json"))
        
        for json_file in json_files:
            original_name = json_file.name
            base_name = json_file.stem
            
            # Always add prefix (including H1)
            new_name = f"{subdir.name}_{original_name}"
            
            label_name_counts[original_name] += 1
            
            target_path = target_labels_dir / new_name
            
            # If target already exists with different content, add counter
            if target_path.exists() and target_path.stat().st_size != json_file.stat().st_size:
                counter = 1
                while target_path.exists():
                    new_name = f"{subdir.name}_{base_name}_{counter}.json"
                    target_path = target_labels_dir / new_name
                    counter += 1
            
            shutil.copy2(json_file, target_path)
            print(f"    Copied: {json_file.name} -> {new_name}")
    
    print(f"\n  Total labels copied: {sum(label_name_counts.values())}")
    
    # Report duplicates
    duplicate_images = {k: v for k, v in image_name_counts.items() if v > 1}
    duplicate_labels = {k: v for k, v in label_name_counts.items() if v > 1}
    
    if duplicate_images:
        print(f"\n[INFO] Found {len(duplicate_images)} duplicate image names (prefixes added):")
        for name, count in sorted(duplicate_images.items())[:10]:
            print(f"  {name}: {count} copies")
        if len(duplicate_images) > 10:
            print(f"  ... and {len(duplicate_images) - 10} more")
    
    if duplicate_labels:
        print(f"\n[INFO] Found {len(duplicate_labels)} duplicate label names (prefixes added):")
        for name, count in sorted(duplicate_labels.items())[:10]:
            print(f"  {name}: {count} copies")
        if len(duplicate_labels) > 10:
            print(f"  ... and {len(duplicate_labels) - 10} more")
    
    print(f"\n[OK] Merge complete!")
    print(f"  Images: {target_images_dir}")
    print(f"  Labels: {target_labels_dir}")


if __name__ == "__main__":
    import sys
    
    source = Path("Dataset_Reorganized")
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    
    target_images = Path("Dataset_Merged/images_all")
    target_labels = Path("Dataset_Merged/labels_all")
    
    if len(sys.argv) > 2:
        base_target = Path(sys.argv[2])
        target_images = base_target / "images_all"
        target_labels = base_target / "labels_all"
    
    print(f"Source: {source}")
    print(f"Target images: {target_images}")
    print(f"Target labels: {target_labels}")
    print()
    
    merge_dataset_folders(source, target_images, target_labels)

