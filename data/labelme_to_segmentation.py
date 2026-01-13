import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def collect_label_map(annotation_dirs: List[Path]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for ann_dir in annotation_dirs:
        # Sort json files for consistent ordering regardless of filesystem
        for json_path in sorted(ann_dir.glob("*.json")):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Support both LabelMe and Label Studio formats
            shapes = data.get("shapes", [])
            if not shapes:
                # Try Label Studio format
                shapes = data.get("labels", [])
            
            for shape in shapes:
                # LabelMe: shape["label"] is string
                # Label Studio: shape["polygonlabels"] is array
                label = shape.get("label")
                if not label:
                    # Try Label Studio format
                    polygon_labels = shape.get("polygonlabels", [])
                    if polygon_labels:
                        label = polygon_labels[0]
                
                if not label:
                    continue
                if label not in labels:
                    # 0 is reserved for background
                    labels[label] = len(labels) + 1
    return labels


def find_annotation_image_pairs(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    annotation_dirs: List[Path] = []
    image_dirs: List[Path] = []
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


def create_mask_for_annotation(
    json_path: Path,
    image_dir: Path,
    masks_dir: Path,
    label_map: Dict[str, int],
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try to find image by JSON filename (ignore imagePath as it may be outdated)
    # JSON: splits/train/labels/M1_frame_021.json
    # Image: splits/train/images/M1_frame_021.png
    base_name = json_path.stem
    
    # Try multiple extensions
    candidates = [
        image_dir / f"{base_name}.png",
        image_dir / f"{base_name}.jpg",
        image_dir / f"{base_name}.PNG",
        image_dir / f"{base_name}.JPG",
    ]
    
    # Also try using imagePath if it's a simple filename
    image_filename = data.get("imagePath")
    if image_filename:
        # Extract just the filename if it's a path
        if "/" in image_filename or "\\" in image_filename:
            image_filename = Path(image_filename).name
        candidates.insert(0, image_dir / image_filename)
    
    img_path = None
    for candidate in candidates:
        if candidate.exists():
            img_path = candidate
            break
    
    if img_path is None:
        # Raise error instead of silent skip so caller can handle it
        raise FileNotFoundError(
            f"Image not found for {json_path.name}. Searched in:\n"
            + "\n".join(f"  - {c}" for c in candidates)
        )

    with Image.open(img_path) as img:
        w, h = img.size

    mask = Image.new("I", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Support both LabelMe and Label Studio formats
    shapes = data.get("shapes", [])
    if not shapes:
        # Try Label Studio format
        shapes = data.get("labels", [])

    for shape in shapes:
        # LabelMe: shape["label"] is string
        # Label Studio / uniform: shape["polygonlabels"] is array
        label = shape.get("label")
        if not label:
            polygon_labels = shape.get("polygonlabels", [])
            if polygon_labels:
                label = polygon_labels[0]

        points = shape.get("points")
        if not label or not points:
            continue

        class_id = label_map.get(label)
        if class_id is None:
            continue

        coord_type = shape.get("coordinate_type") or "auto"
        orig_w = shape.get("original_width", w)
        orig_h = shape.get("original_height", h)

        # Heuristic: infer coordinate type when missing
        if coord_type not in ("actual", "normalized"):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            max_x, max_y = max(xs), max(ys)
            min_x, min_y = min(xs), min(ys)
            # Case 1: very small fractions (0-1) → normalized fractions
            if max_x <= 1.0 and max_y <= 1.0:
                coord_type = "normalized"
            # Case 2: 0-100 percentages while image is larger than 120px
            elif max_x <= 110.0 and max_y <= 110.0 and (w > 120 or h > 120):
                coord_type = "normalized"
            else:
                coord_type = "actual"

        polygon: list[tuple[float, float]] = []
        if coord_type == "normalized":
            # Points are 0-100 (or 0-1) percentages -> convert to pixels using target image size
            denom_x = 100.0 if max(p[0] for p in points) > 1.0 else 1.0
            denom_y = 100.0 if max(p[1] for p in points) > 1.0 else 1.0
            for x, y in points:
                polygon.append((float(x) / denom_x * w, float(y) / denom_y * h))
        else:
            # Actual pixel coordinates; scale if original dims differ from current image
            scale_x = w / orig_w if orig_w else 1.0
            scale_y = h / orig_h if orig_h else 1.0
            for x, y in points:
                polygon.append((float(x) * scale_x, float(y) * scale_y))

        if len(polygon) >= 3:
            draw.polygon(polygon, outline=class_id, fill=class_id)

    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f"{json_path.stem}.png"
    # Convert to uint8 before saving to keep file size reasonable
    arr = np.array(mask, dtype=np.uint8)
    Image.fromarray(arr).save(mask_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to semantic segmentation masks."
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
        default="labels/segmentation",
        help="Output directory for mask images.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_dir)

    ann_dirs, img_dirs = find_annotation_image_pairs(dataset_root)
    if not ann_dirs:
        raise SystemExit(f"No *_Annotations folders found under {dataset_root}")

    label_map = collect_label_map(ann_dirs)
    print("Detected classes (background=0):")
    for name, cid in label_map.items():
        print(f"  {cid}: {name}")

    for ann_dir, img_dir in zip(ann_dirs, img_dirs):
        split_name = ann_dir.name.replace("_Annotations", "")
        split_masks_dir = output_root / split_name
        print(f"Processing {ann_dir} -> {split_masks_dir}")
        for json_path in sorted(ann_dir.glob("*.json")):
            create_mask_for_annotation(json_path, img_dir, split_masks_dir, label_map)

    output_root.mkdir(parents=True, exist_ok=True)
    names_path = output_root / "classes.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("0 background\n")
        for name, cid in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{cid} {name}\n")

    print(f"Done. Masks written under {output_root}")


if __name__ == "__main__":
    main()
