"""
Uniform JSON Format Converter
Converts between different annotation formats to a uniform Label Studio-style format.
Handles both LabelMe format (actual coordinates) and Label Studio format (normalized coordinates).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2


def detect_json_format(data: dict) -> str:
    """
    Detect the format of the JSON annotation file.
    
    Returns:
        'labelme' - Format 1 (has 'shapes' key, uses actual coordinates)
        'labelstudio' - Format 2 (has 'labels' key, may use normalized coordinates)
        'uniform' - Already in uniform format (has 'labels' and 'coordinate_type')
        'unknown' - Cannot determine format
    """
    if "shapes" in data:
        return "labelme"
    elif "labels" in data:
        # Check if it has coordinate_type field (uniform format)
        if data.get("labels") and any("coordinate_type" in label for label in data["labels"]):
            return "uniform"
        return "labelstudio"
    return "unknown"


def get_image_dimensions(json_path: Path, data: dict) -> Tuple[int, int]:
    """
    Get image dimensions from various sources.
    
    Priority:
    1. From JSON data (original_width, original_height)
    2. From imageWidth, imageHeight fields
    3. From actual image file
    
    Returns (width, height)
    """
    # Try to get from labels (Label Studio format)
    if "labels" in data and data["labels"]:
        first_label = data["labels"][0]
        if "original_width" in first_label and "original_height" in first_label:
            return first_label["original_width"], first_label["original_height"]
    
    # Try to get from top-level fields (LabelMe format)
    if "imageWidth" in data and "imageHeight" in data:
        return data["imageWidth"], data["imageHeight"]
    
    # Try to load the actual image
    image_path = _find_image_for_json(json_path, data)
    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    
    raise ValueError(f"Cannot determine image dimensions for {json_path}")


def _find_image_for_json(json_path: Path, data: dict) -> Path:
    """Find the corresponding image file for a JSON annotation."""
    # Try imagePath from LabelMe format
    if "imagePath" in data:
        img_rel_path = data["imagePath"]
        # Handle relative paths like "..\\H1\\frame_004.png"
        img_path = (json_path.parent / img_rel_path).resolve()
        if img_path.exists():
            return img_path
    
    # Try image field from Label Studio format
    if "image" in data:
        img_rel_path = data["image"]
        # Extract filename from path like "/data/upload/2/c217044a-frame_001.png"
        img_name = Path(img_rel_path).name
        # Remove prefix like "c217044a-" if present
        if "-" in img_name:
            img_name = img_name.split("-", 1)[1]
        
        # Search in common locations
        dataset_root = json_path.parent.parent
        for search_dir in [json_path.parent, dataset_root / "images_all", dataset_root / "images"]:
            if search_dir.exists():
                img_path = search_dir / img_name
                if img_path.exists():
                    return img_path
    
    # Try same name as JSON
    img_name = json_path.stem
    for ext in [".png", ".jpg", ".jpeg"]:
        img_path = json_path.parent / f"{img_name}{ext}"
        if img_path.exists():
            return img_path
    
    # Search in sibling directories
    dataset_root = json_path.parent.parent
    for search_dir in [dataset_root / "images_all", dataset_root / "images"]:
        if search_dir.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = search_dir / f"{img_name}{ext}"
                if img_path.exists():
                    return img_path
    
    return None


def convert_labelme_to_uniform(data: dict, img_width: int, img_height: int, json_path: Path) -> dict:
    """
    Convert LabelMe format (Format 1) to uniform format.
    LabelMe uses actual coordinates already, so we just restructure.
    """
    # Find image path
    image_path = _find_image_for_json(json_path, data)
    if image_path:
        # Make relative to JSON location
        try:
            img_rel = image_path.relative_to(json_path.parent)
        except ValueError:
            img_rel = image_path.name
    else:
        # Use imagePath from data or construct from JSON name
        img_rel = data.get("imagePath", f"{json_path.stem}.png")
    
    labels = []
    for shape in data.get("shapes", []):
        label_name = shape.get("label", "Unknown")
        points = shape.get("points", [])
        
        if not points:
            continue
        
        label_entry = {
            "points": points,  # Already in actual coordinates
            "closed": True,
            "polygonlabels": [label_name],
            "original_width": img_width,
            "original_height": img_height,
            "coordinate_type": "actual"
        }
        labels.append(label_entry)
    
    return {
        "image": str(img_rel),
        "labels": labels
    }


def convert_labelstudio_to_uniform(data: dict, img_width: int, img_height: int, json_path: Path) -> dict:
    """
    Convert Label Studio format (Format 2) to uniform format.
    Label Studio uses normalized coordinates (0-100), so we convert to actual.
    """
    # Find image path
    image_path = _find_image_for_json(json_path, data)
    if image_path:
        try:
            img_rel = image_path.relative_to(json_path.parent)
        except ValueError:
            img_rel = image_path.name
    else:
        img_rel = data.get("image", f"{json_path.stem}.png")
    
    labels = []
    for label in data.get("labels", []):
        points_normalized = label.get("points", [])
        polygon_labels = label.get("polygonlabels", ["Unknown"])
        
        if not points_normalized:
            continue
        
        # Convert normalized (0-100) to actual pixel coordinates
        points_actual = [
            [p[0] / 100.0 * img_width, p[1] / 100.0 * img_height]
            for p in points_normalized
        ]
        
        label_entry = {
            "points": points_actual,
            "closed": label.get("closed", True),
            "polygonlabels": polygon_labels,
            "original_width": img_width,
            "original_height": img_height,
            "coordinate_type": "actual"
        }
        labels.append(label_entry)
    
    return {
        "image": str(img_rel),
        "labels": labels
    }


def convert_to_uniform_format(json_path: Path, backup: bool = True) -> Tuple[bool, str]:
    """
    Convert any supported format to uniform format with actual coordinates.
    
    Returns:
        (success: bool, message: str)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        format_type = detect_json_format(data)
        
        if format_type == "uniform":
            # Check if already using actual coordinates
            if data.get("labels") and data["labels"][0].get("coordinate_type") == "actual":
                return True, "Already in uniform format with actual coordinates"
            # If normalized, we'll convert it
        
        # Get image dimensions
        img_width, img_height = get_image_dimensions(json_path, data)
        
        # Convert based on format
        if format_type == "labelme":
            uniform_data = convert_labelme_to_uniform(data, img_width, img_height, json_path)
        elif format_type == "labelstudio" or format_type == "uniform":
            uniform_data = convert_labelstudio_to_uniform(data, img_width, img_height, json_path)
        else:
            return False, f"Unknown format: {format_type}"
        
        # Backup if requested
        if backup:
            backup_path = json_path.with_suffix(".json.backup")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Write uniform format
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(uniform_data, f, indent=2, ensure_ascii=False)
        
        return True, f"Converted from {format_type} to uniform format"
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def detect_coordinate_type(json_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Detect whether a uniform JSON uses actual or normalized coordinates.
    
    Returns:
        (coordinate_type: str, info: dict)
        coordinate_type can be: 'actual', 'normalized', 'mixed', 'unknown'
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "labels" not in data or not data["labels"]:
            return "unknown", {"error": "No labels found"}
        
        # Check coordinate_type field if present
        coord_types = set()
        for label in data["labels"]:
            if "coordinate_type" in label:
                coord_types.add(label["coordinate_type"])
        
        if len(coord_types) == 1:
            return list(coord_types)[0], {"source": "coordinate_type field"}
        elif len(coord_types) > 1:
            return "mixed", {"types": list(coord_types)}
        
        # Try to infer from point values
        img_width = data["labels"][0].get("original_width", 0)
        img_height = data["labels"][0].get("original_height", 0)
        
        if img_width == 0 or img_height == 0:
            return "unknown", {"error": "No image dimensions"}
        
        # Sample some points
        sample_points = []
        for label in data["labels"][:3]:  # Check first 3 labels
            points = label.get("points", [])
            sample_points.extend(points[:5])  # First 5 points per label
        
        if not sample_points:
            return "unknown", {"error": "No points found"}
        
        # Check if all points are in range [0, 100] (normalized)
        all_in_0_100 = all(0 <= p[0] <= 100 and 0 <= p[1] <= 100 for p in sample_points)
        # Check if any point exceeds 100 (likely actual)
        any_exceeds_100 = any(p[0] > 100 or p[1] > 100 for p in sample_points)
        
        if any_exceeds_100:
            return "actual", {"source": "inferred from point values", "sample_points": sample_points[:3]}
        elif all_in_0_100:
            return "normalized", {"source": "inferred from point values", "sample_points": sample_points[:3]}
        else:
            return "unknown", {"sample_points": sample_points[:3]}
    
    except Exception as e:
        return "unknown", {"error": str(e)}


def convert_coordinates(json_path: Path, target_type: str, backup: bool = True) -> Tuple[bool, str]:
    """
    Convert coordinates between actual and normalized.
    
    Args:
        json_path: Path to JSON file
        target_type: 'actual' or 'normalized'
        backup: Whether to create backup
    
    Returns:
        (success: bool, message: str)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "labels" not in data or not data["labels"]:
            return False, "No labels found in JSON"
        
        # Detect current type
        current_type, info = detect_coordinate_type(json_path)
        
        if current_type == target_type:
            return True, f"Already in {target_type} format"
        
        if current_type == "unknown":
            return False, f"Cannot determine current coordinate type: {info}"
        
        if current_type == "mixed":
            return False, "Mixed coordinate types detected, cannot convert"
        
        # Backup if requested
        if backup:
            backup_path = json_path.with_suffix(".json.backup")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Convert each label
        converted_labels = []
        for label in data["labels"]:
            img_width = label.get("original_width")
            img_height = label.get("original_height")
            
            if not img_width or not img_height:
                return False, "Missing image dimensions in label"
            
            points = label.get("points", [])
            
            if target_type == "normalized" and current_type == "actual":
                # Convert actual to normalized (0-100)
                new_points = [
                    [p[0] / img_width * 100.0, p[1] / img_height * 100.0]
                    for p in points
                ]
            elif target_type == "actual" and current_type == "normalized":
                # Convert normalized (0-100) to actual
                new_points = [
                    [p[0] / 100.0 * img_width, p[1] / 100.0 * img_height]
                    for p in points
                ]
            else:
                return False, f"Cannot convert from {current_type} to {target_type}"
            
            converted_label = label.copy()
            converted_label["points"] = new_points
            converted_label["coordinate_type"] = target_type
            converted_labels.append(converted_label)
        
        # Update data
        data["labels"] = converted_labels
        
        # Write back
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True, f"Converted from {current_type} to {target_type}"
    
    except Exception as e:
        return False, f"Error: {str(e)}"


def uniform_all_jsons(dataset_root: Path, backup: bool = True) -> Dict[str, Any]:
    """
    Convert all JSON files in dataset to uniform format with actual coordinates.
    
    Returns statistics dictionary.
    """
    stats = {
        "total": 0,
        "converted": 0,
        "already_uniform": 0,
        "errors": [],
        "by_format": {"labelme": 0, "labelstudio": 0, "uniform": 0, "unknown": 0}
    }
    
    # Find all JSON files
    json_files = list(dataset_root.rglob("*.json"))
    stats["total"] = len(json_files)
    
    for json_path in json_files:
        # Skip backup files
        if json_path.suffix == ".backup" or ".backup" in json_path.suffixes:
            stats["total"] -= 1
            continue
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            format_type = detect_json_format(data)
            stats["by_format"][format_type] = stats["by_format"].get(format_type, 0) + 1
            
            success, message = convert_to_uniform_format(json_path, backup=backup)
            
            if success:
                if "Already in uniform" in message:
                    stats["already_uniform"] += 1
                else:
                    stats["converted"] += 1
            else:
                stats["errors"].append(f"{json_path.name}: {message}")
        
        except Exception as e:
            stats["errors"].append(f"{json_path.name}: {str(e)}")
    
    return stats


def analyze_dataset_coordinates(dataset_root: Path) -> Dict[str, Any]:
    """
    Analyze all JSON files to detect coordinate types.
    
    Returns analysis dictionary.
    """
    analysis = {
        "total": 0,
        "actual": 0,
        "normalized": 0,
        "mixed": 0,
        "unknown": 0,
        "samples": [],
        "errors": []
    }
    
    json_files = list(dataset_root.rglob("*.json"))
    
    for json_path in json_files:
        # Skip backup files
        if json_path.suffix == ".backup" or ".backup" in json_path.suffixes:
            continue
        
        analysis["total"] += 1
        
        try:
            coord_type, info = detect_coordinate_type(json_path)
            analysis[coord_type] = analysis.get(coord_type, 0) + 1
            
            # Store some samples
            if len(analysis["samples"]) < 5:
                analysis["samples"].append({
                    "file": json_path.name,
                    "type": coord_type,
                    "info": info
                })
        
        except Exception as e:
            analysis["errors"].append(f"{json_path.name}: {str(e)}")
    
    return analysis


def batch_convert_coordinates(dataset_root: Path, target_type: str, backup: bool = True) -> Dict[str, Any]:
    """
    Convert all JSON files to specified coordinate type.
    
    Returns statistics dictionary.
    """
    stats = {
        "total": 0,
        "converted": 0,
        "already_target": 0,
        "errors": []
    }
    
    json_files = list(dataset_root.rglob("*.json"))
    
    for json_path in json_files:
        # Skip backup files
        if json_path.suffix == ".backup" or ".backup" in json_path.suffixes:
            continue
        
        stats["total"] += 1
        
        try:
            success, message = convert_coordinates(json_path, target_type, backup=backup)
            
            if success:
                if "Already in" in message:
                    stats["already_target"] += 1
                else:
                    stats["converted"] += 1
            else:
                stats["errors"].append(f"{json_path.name}: {message}")
        
        except Exception as e:
            stats["errors"].append(f"{json_path.name}: {str(e)}")
    
    return stats


def polygon_to_bbox(points: List[List[float]]) -> List[List[float]]:
    """
    Convert polygon points to bounding box.
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        Bounding box as [[x_min, y_min], [x_max, y_max]]
    """
    if not points or len(points) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    return [[x_min, y_min], [x_max, y_max]]


def convert_json_polygons_to_bboxes(json_path: Path, backup: bool = True) -> Tuple[bool, str]:
    """
    Convert all polygons in a JSON file to bounding boxes.
    
    Returns:
        (success, message)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to read JSON: {str(e)}"
    
    # Create backup if requested
    if backup:
        backup_path = json_path.with_suffix(".json.backup")
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return False, f"Failed to create backup: {str(e)}"
    
    modified = False
    polygon_count = 0
    
    # Handle shapes format (LabelMe-style)
    if "shapes" in data:
        for shape in data["shapes"]:
            points = shape.get("points", [])
            if len(points) >= 3:
                try:
                    bbox = polygon_to_bbox(points)
                    shape["points"] = bbox
                    shape["shape_type"] = "rectangle"  # Mark as bounding box
                    modified = True
                    polygon_count += 1
                except Exception as e:
                    return False, f"Failed to convert polygon in shapes: {str(e)}"
    
    # Handle labels format (Label Studio-style / Uniform format)
    if "labels" in data:
        for label in data["labels"]:
            points = label.get("points", [])
            if len(points) >= 3:
                try:
                    bbox = polygon_to_bbox(points)
                    label["points"] = bbox
                    # Keep coordinate_type if it exists
                    modified = True
                    polygon_count += 1
                except Exception as e:
                    return False, f"Failed to convert polygon in labels: {str(e)}"
    
    if not modified:
        return True, "No polygons found or already in bbox format"
    
    # Save the modified data
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return False, f"Failed to save JSON: {str(e)}"
    
    return True, f"Converted {polygon_count} polygon(s) to bounding box(es)"


def convert_polygons_to_bboxes(dataset_root: Path, backup: bool = True) -> Dict[str, Any]:
    """
    Convert all polygon annotations to bounding boxes in dataset.
    
    Args:
        dataset_root: Root directory of dataset
        backup: Whether to create backup files
    
    Returns:
        Statistics dictionary with conversion results
    """
    stats = {
        "total": 0,
        "converted_files": 0,
        "converted_polygons": 0,
        "skipped": 0,
        "errors": []
    }
    
    # Look for JSON files in labels_all directory
    labels_dir = dataset_root / "labels_all"
    if not labels_dir.exists():
        # Fallback to searching entire dataset
        json_files = list(dataset_root.rglob("*.json"))
    else:
        json_files = list(labels_dir.glob("*.json"))
    
    for json_path in json_files:
        # Skip backup files
        if json_path.suffix == ".backup" or ".backup" in json_path.suffixes:
            continue
        
        stats["total"] += 1
        
        try:
            success, message = convert_json_polygons_to_bboxes(json_path, backup=backup)
            
            if success:
                if "No polygons found" in message:
                    stats["skipped"] += 1
                else:
                    stats["converted_files"] += 1
                    # Extract polygon count from message
                    try:
                        count = int(message.split()[1])
                        stats["converted_polygons"] += count
                    except:
                        stats["converted_polygons"] += 1
            else:
                stats["errors"].append(f"{json_path.name}: {message}")
        
        except Exception as e:
            stats["errors"].append(f"{json_path.name}: {str(e)}")
    
    return stats
