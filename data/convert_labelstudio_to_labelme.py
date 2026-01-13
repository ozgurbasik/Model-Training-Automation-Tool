import json
import base64
from pathlib import Path
from typing import Dict, Any

def convert_labelstudio_to_labelme(labelstudio_data: Dict[str, Any], json_file_path: Path) -> Dict[str, Any]:
    """Convert Label Studio / Uniform format to LabelMe format.
    
    Both formats use actual pixel coordinates.
    This function just restructures the data format without coordinate conversion.
    """
    
    # Extract image path
    image_path = labelstudio_data.get("image", "")
    # Remove path prefix if exists, keep just filename
    if "/" in image_path or "\\" in image_path:
        image_path = Path(image_path).name
    
    # Get image dimensions from first label if available
    image_width = None
    image_height = None
    if "labels" in labelstudio_data and len(labelstudio_data["labels"]) > 0:
        first_label = labelstudio_data["labels"][0]
        image_width = first_label.get("original_width")
        image_height = first_label.get("original_height")
    
    # Convert labels to shapes
    shapes = []
    for label_obj in labelstudio_data.get("labels", []):
        points = label_obj.get("points", [])
        polygonlabels = label_obj.get("polygonlabels", [])
        coord_type = label_obj.get("coordinate_type", "actual")
        
        if not points or not polygonlabels:
            continue
        
        # Get label name (take first if array)
        label_name = polygonlabels[0] if isinstance(polygonlabels, list) else polygonlabels
        
        # Get dimensions for this specific label
        W = label_obj.get("original_width", image_width)
        H = label_obj.get("original_height", image_height)
        
        # Handle coordinates based on type
        converted_points = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                x, y = float(point[0]), float(point[1])
                
                if coord_type == "actual":
                    # Already in pixels - use directly
                    converted_points.append([x, y])
                else:
                    # Normalized (0-100) - convert to pixels
                    if W is not None and H is not None:
                        x_pixel = (x / 100.0) * W
                        y_pixel = (y / 100.0) * H
                        converted_points.append([x_pixel, y_pixel])
                    else:
                        # Fallback: keep as-is if dimensions unknown
                        converted_points.append([x, y])
            else:
                # Skip invalid points
                continue
        
        if len(converted_points) < 3:
            continue
        
        # Create LabelMe shape
        shape = {
            "label": label_name,
            "points": converted_points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)
    
    # Create LabelMe format
    labelme_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,  # We don't have image data, leave as None
    }
    
    # Add image dimensions if available
    if image_width is not None:
        labelme_data["imageWidth"] = image_width
    if image_height is not None:
        labelme_data["imageHeight"] = image_height
    
    return labelme_data


def convert_all_labelstudio_files(labels_dir: Path, backup: bool = True):
    """Convert all Label Studio format files to LabelMe format."""
    
    # Get all JSON files
    json_files = list(labels_dir.glob("*.json"))
    
    converted_count = 0
    error_count = 0
    
    print(f"Found {len(json_files)} JSON files")
    print("Converting Label Studio format to LabelMe format...\n")
    
    for json_file in json_files:
        try:
            # Read file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's Label Studio format
            if "image" in data and "labels" in data and "version" not in data:
                # Backup original if requested
                if backup:
                    backup_path = json_file.with_suffix('.json.backup')
                    if not backup_path.exists():
                        import shutil
                        shutil.copy2(json_file, backup_path)
                
                # Convert
                labelme_data = convert_labelstudio_to_labelme(data, json_file)
                
                # Write converted data
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                converted_count += 1
                if converted_count % 10 == 0:
                    print(f"  Converted {converted_count} files...")
        
        except Exception as e:
            error_count += 1
            print(f"Error converting {json_file.name}: {e}")
    
    print(f"\n[OK] Conversion complete!")
    print(f"  Converted: {converted_count} files")
    print(f"  Errors: {error_count} files")
    
    if backup:
        print(f"\n[INFO] Original files backed up with .backup extension")


if __name__ == "__main__":
    import sys
    
    labels_dir = Path("Dataset_Merged/labels")
    if len(sys.argv) > 1:
        labels_dir = Path(sys.argv[1])
    
    if not labels_dir.exists():
        print(f"Error: {labels_dir} not found!")
        sys.exit(1)
    
    convert_all_labelstudio_files(labels_dir, backup=True)

