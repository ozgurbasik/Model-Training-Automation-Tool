"""
Reorganize dataset from original format (H1/, H1_Annotations/, etc.)
to merged format (images_all/, labels_all/).

Files are renamed with prefix (H1_frame_001.png, H1_frame_001.json, etc.)
to avoid collisions when consolidating from multiple folders.
"""

import argparse
import shutil
from pathlib import Path
from typing import Tuple, List


def find_image_annotation_pairs(dataset_root: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find all image folders and their corresponding annotation folders.
    Returns list of (image_folder, annotation_folder, prefix) tuples.
    """
    pairs = []
    
    for item in sorted(dataset_root.iterdir()):
        if not item.is_dir():
            continue
        
        # Look for folders like H1, H2, M1, M2, etc. (not ending with _Annotations)
        if item.name.endswith("_Annotations"):
            continue
        
        # Check if corresponding *_Annotations folder exists
        ann_folder = dataset_root / f"{item.name}_Annotations"
        if ann_folder.exists() and ann_folder.is_dir():
            pairs.append((item, ann_folder, item.name))
    
    return pairs


def reorganize_dataset(dataset_root: Path, output_dir: Path = None, dry_run: bool = False) -> None:
    """
    Reorganize dataset to merged format with prefixed filenames.
    
    Args:
        dataset_root: Root directory containing H1, H1_Annotations, etc.
        output_dir: Output directory for images_all and labels_all (default: dataset_root)
        dry_run: If True, only print what would be done without copying
    """
    if output_dir is None:
        output_dir = dataset_root
    
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # Find all image-annotation pairs
    pairs = find_image_annotation_pairs(dataset_root)
    
    if not pairs:
        print(f"❌ No image-annotation pairs found in {dataset_root}")
        print("   Expected format: H1/, H1_Annotations/, H2/, H2_Annotations/, etc.")
        return
    
    print(f"✅ Found {len(pairs)} image-annotation folder pairs:")
    for img_dir, ann_dir, prefix in pairs:
        print(f"   {prefix}: {img_dir.name} + {ann_dir.name}")
    
    # Create output directories
    images_all = output_dir / "images_all"
    labels_all = output_dir / "labels_all"
    
    if not dry_run:
        images_all.mkdir(parents=True, exist_ok=True)
        labels_all.mkdir(parents=True, exist_ok=True)
    
    image_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
    label_extensions = [".json", ".txt"]
    
    total_images = 0
    total_labels = 0
    duplicates = []
    
    # Process each pair
    for img_dir, ann_dir, prefix in pairs:
        print(f"\n📁 Processing {prefix}...")
        
        # Copy images with prefix
        image_count = 0
        for img_file in img_dir.iterdir():
            if img_file.is_file() and any(str(img_file).endswith(ext) for ext in image_extensions):
                # Create prefixed filename
                new_name = f"{prefix}_{img_file.name}"
                dst_path = images_all / new_name
                
                # Check for duplicates
                if dst_path.exists():
                    duplicates.append(f"Image: {new_name}")
                    print(f"   ⚠️  Skipping duplicate image: {new_name}")
                    continue
                
                if not dry_run:
                    shutil.copy2(img_file, dst_path)
                
                image_count += 1
                total_images += 1
                print(f"   ✓ Image: {new_name}")
        
        # Copy labels with prefix
        label_count = 0
        for label_file in ann_dir.iterdir():
            if label_file.is_file() and any(str(label_file).endswith(ext) for ext in label_extensions):
                # Create prefixed filename
                new_name = f"{prefix}_{label_file.name}"
                dst_path = labels_all / new_name
                
                # Check for duplicates
                if dst_path.exists():
                    duplicates.append(f"Label: {new_name}")
                    print(f"   ⚠️  Skipping duplicate label: {new_name}")
                    continue
                
                if not dry_run:
                    shutil.copy2(label_file, dst_path)
                
                label_count += 1
                total_labels += 1
                print(f"   ✓ Label: {new_name}")
        
        print(f"   {prefix}: {image_count} images, {label_count} labels")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Reorganization Summary")
    print(f"{'='*60}")
    print(f"Total images copied: {total_images}")
    print(f"Total labels copied: {total_labels}")
    
    if duplicates:
        print(f"\n⚠️  Duplicates found ({len(duplicates)}):")
        for dup in duplicates[:10]:
            print(f"   - {dup}")
        if len(duplicates) > 10:
            print(f"   ... and {len(duplicates) - 10} more")
    
    if not dry_run:
        print(f"\n✅ Reorganization complete!")
        print(f"   Images: {images_all}")
        print(f"   Labels: {labels_all}")
        
        # Delete original folders after successful conversion
        print(f"\n🗑️  Deleting original folders...")
        for img_dir, ann_dir, prefix in pairs:
            try:
                shutil.rmtree(img_dir)
                print(f"   ✓ Deleted: {img_dir.name}")
            except Exception as e:
                print(f"   ❌ Failed to delete {img_dir.name}: {e}")
            
            try:
                shutil.rmtree(ann_dir)
                print(f"   ✓ Deleted: {ann_dir.name}")
            except Exception as e:
                print(f"   ❌ Failed to delete {ann_dir.name}: {e}")
        
        print(f"\n✅ Original folders deleted. Only images_all/ and labels_all/ remain.")
    else:
        print(f"\n📝 Dry run complete. Use without --dry-run to actually copy files and delete originals.")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize dataset from H1/H1_Annotations format to images_all/labels_all format"
    )
    
    # Find default dataset path relative to script location
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent  # Go up from data/ to repo root
    default_dataset = repo_root / "DataSet" / "RcCArDataset"
    
    parser.add_argument(
        "dataset_root",
        type=str,
        nargs="?",  # Make it optional
        default=str(default_dataset),
        help=f"Root directory containing H1, H1_Annotations, etc. (default: {default_dataset})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for images_all and labels_all (default: same as dataset_root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually copying files",
    )
    
    args = parser.parse_args()
    
    reorganize_dataset(
        Path(args.dataset_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
