import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def polygon_to_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max


def collect_label_map(annotation_dirs: List[Path]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for ann_dir in annotation_dirs:
        # Sort json files for consistent ordering regardless of filesystem
        for json_path in sorted(ann_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle shapes format (LabelMe)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label is None:
                    continue
                if label not in labels:
                    labels[label] = len(labels)
            
            # Handle labels format (uniform)
            for lbl in data.get("labels", []):
                label_list = lbl.get("polygonlabels", [])
                label = label_list[0] if label_list else None
                if label is None:
                    continue
                if label not in labels:
                    labels[label] = len(labels)
    return labels


def process_annotation(
    json_path: Path,
    image_dir: Path,
    labels_dir: Path,
    label_map: Dict[str, int],
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try both "imagePath" (LabelMe) and "image" (uniform format)
    image_filename = data.get("imagePath") or data.get("image")
    if image_filename is None:
        # Fallback: assume same stem with .png
        image_filename = json_path.stem + ".png"

    img_path = image_dir / image_filename
    if not img_path.exists():
        # Try jpg as fallback
        jpg_path = image_dir / (json_path.stem + ".jpg")
        if jpg_path.exists():
            img_path = jpg_path
        else:
            print(f"[WARN] Image not found for {json_path}")
            return

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Failed to read image {img_path}")
        return
    h, w = img.shape[:2]

    yolo_lines: List[str] = []
    
    # Process shapes format (LabelMe)
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")
        if label is None or not points or len(points) < 2:
            continue
        class_id = label_map.get(label)
        if class_id is None:
            continue
        
        # Handle both bbox (2 points) and polygon (3+ points)
        if len(points) == 2:
            # Already a bounding box [[x_min, y_min], [x_max, y_max]]
            x_min, y_min = points[0]
            x_max, y_max = points[1]
        else:
            # Polygon - convert to bbox
            x_min, y_min, x_max, y_max = polygon_to_bbox(points)

        # Clip to image
        x_min = max(0.0, min(float(x_min), float(w - 1)))
        x_max = max(0.0, min(float(x_max), float(w - 1)))
        y_min = max(0.0, min(float(y_min), float(h - 1)))
        y_max = max(0.0, min(float(y_max), float(h - 1)))
        if x_max <= x_min or y_max <= y_min:
            continue

        cx = (x_min + x_max) / 2.0 / w
        cy = (y_min + y_max) / 2.0 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    
    # Process labels format (uniform)
    for lbl in data.get("labels", []):
        label_list = lbl.get("polygonlabels", [])
        label = label_list[0] if label_list else None
        points = lbl.get("points", [])
        if label is None or not points or len(points) < 2:
            continue
        class_id = label_map.get(label)
        if class_id is None:
            continue
        
        # Get coordinate type and dimensions
        coord_type = lbl.get("coordinate_type", "normalized")
        orig_width = lbl.get("original_width", w)
        orig_height = lbl.get("original_height", h)
        
        # Convert to actual pixels if needed
        if coord_type == "actual":
            # Scale if dimensions don't match
            if orig_width != w or orig_height != h:
                scale_x = w / orig_width if orig_width else 1.0
                scale_y = h / orig_height if orig_height else 1.0
                points = [[p[0] * scale_x, p[1] * scale_y] for p in points]
        else:
            # Normalized (0-100) - convert to pixels
            points = [[p[0] / 100.0 * w, p[1] / 100.0 * h] for p in points]
        
        # Handle both bbox (2 points) and polygon (3+ points)
        if len(points) == 2:
            # Already a bounding box [[x_min, y_min], [x_max, y_max]]
            x_min, y_min = points[0]
            x_max, y_max = points[1]
        else:
            # Polygon - convert to bbox
            x_min, y_min, x_max, y_max = polygon_to_bbox(points)

        # Clip to image
        x_min = max(0.0, min(float(x_min), float(w - 1)))
        x_max = max(0.0, min(float(x_max), float(w - 1)))
        y_min = max(0.0, min(float(y_min), float(h - 1)))
        y_max = max(0.0, min(float(y_max), float(h - 1)))
        if x_max <= x_min or y_max <= y_min:
            continue

        cx = (x_min + x_max) / 2.0 / w
        cy = (y_min + y_max) / 2.0 / h
        bw = (x_max - x_min) / w
        bh = (y_max - y_min) / h
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not yolo_lines:
        return

    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{json_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


def find_annotation_image_pairs(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find all *Annotations folders and corresponding image folders.
    Also handles images_all/labels_all structure.
    """
    annotation_dirs: List[Path] = []
    image_dirs: List[Path] = []
    
    # Check for images_all/labels_all structure first
    labels_all = dataset_root / "labels_all"
    images_all = dataset_root / "images_all"
    if labels_all.is_dir() and images_all.is_dir():
        annotation_dirs.append(labels_all)
        image_dirs.append(images_all)
        return annotation_dirs, image_dirs
    
    # Fallback to X_Annotations/X structure
    for child in dataset_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith("_Annotations"):
            base_name = child.name.replace("_Annotations", "")
            img_dir = dataset_root / base_name
            if img_dir.is_dir():
                annotation_dirs.append(child)
                image_dirs.append(img_dir)
    return annotation_dirs, image_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to YOLO detection format."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="Dataset",
        help="Root directory containing H1, H1_Annotations, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="labels/detection",
        help="Output directory for YOLO label files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_dir)

    ann_dirs, img_dirs = find_annotation_image_pairs(dataset_root)
    if not ann_dirs:
        raise SystemExit(f"No *_Annotations folders found under {dataset_root}")

    label_map = collect_label_map(ann_dirs)
    print("Detected classes:")
    for name, cid in label_map.items():
        print(f"  {cid}: {name}")

    for ann_dir, img_dir in zip(ann_dirs, img_dirs):
        split_name = ann_dir.name.replace("_Annotations", "")
        # If caller already points to labels_all, don't nest another labels_all
        if split_name == "labels_all" and output_root.name == "labels_all":
            split_output_dir = output_root
        else:
            split_output_dir = output_root / split_name
        print(f"Processing {ann_dir} -> {split_output_dir}")
        json_files = sorted(ann_dir.glob("*.json"))
        for json_path in json_files:
            process_annotation(json_path, img_dir, split_output_dir, label_map)

    # Save label map
    names_path = output_root / "classes.txt"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(names_path, "w", encoding="utf-8") as f:
        for name, cid in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{cid} {name}\n")

    print(f"Done. Labels written under {output_root}")


if __name__ == "__main__":
    main()


