import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

# Support both package imports (when run via Streamlit) and direct script execution
try:
    from data.validate_dataset import find_annotation_image_pairs
except ImportError:  # pragma: no cover - runtime safety for mis-set CWD
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (parent of data)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from data.validate_dataset import find_annotation_image_pairs


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def _resolve_image_path(image_dir: Path, image_hint: str, json_stem: str) -> Path | None:
    """Best-effort image lookup for a LabelMe annotation."""
    candidates: List[Path] = []

    if image_hint:
        hint_name = Path(image_hint).name
        candidates.append(image_dir / hint_name)
        # Also search recursively if direct join fails
        candidates.extend(image_dir.rglob(hint_name))

    # Try common extensions using the JSON stem
    for ext in IMAGE_EXTENSIONS:
        candidates.append(image_dir / f"{json_stem}{ext}")
        candidates.extend(image_dir.rglob(f"{json_stem}{ext}"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _labelme_shape_to_labelstudio_entry(shape: Dict[str, Any], width: int, height: int) -> Dict[str, Any] | None:
    """Convert a LabelMe shape to Label Studio format.
    
    LabelMe stores percentage coordinates: [x_percent, y_percent] (0-100)
    Label Studio stores absolute pixel coordinates: [x_pixel, y_pixel] where:
        x_pixel = (x_percent / 100.0) * width
        y_pixel = (y_percent / 100.0) * height
    """
    label = shape.get("label")
    points: List[List[float]] = shape.get("points", [])
    if not label or not points or width <= 0 or height <= 0:
        return None

    scaled_points: List[List[float]] = []
    for x, y in points:
        # Convert percentage (0-100) to absolute pixels
        scaled_points.append([
            (float(x) / 100.0) * float(width),
            (float(y) / 100.0) * float(height),
        ])

    return {
        "points": scaled_points,
        "closed": True,
        "polygonlabels": [label],
        "original_width": width,
        "original_height": height,
    }


def convert_labelme_json_to_labelstudio(labelme_data: Dict[str, Any], image_path: Path | None) -> Dict[str, Any] | None:
    """Convert a single LabelMe JSON payload to Label Studio-style dict."""
    image_width = labelme_data.get("imageWidth")
    image_height = labelme_data.get("imageHeight")

    if (image_width is None or image_height is None) and image_path is not None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        image_height, image_width = img.shape[:2]

    if image_width is None or image_height is None:
        return None

    shapes = labelme_data.get("shapes", [])
    entries: List[Dict[str, Any]] = []
    for shape in shapes:
        entry = _labelme_shape_to_labelstudio_entry(shape, int(image_width), int(image_height))
        if entry:
            entries.append(entry)

    image_field = labelme_data.get("imagePath") or labelme_data.get("image") or ""
    if image_field:
        image_field = Path(image_field).name

    return {
        "image": image_field,
        "labels": entries,
    }


def convert_dataset_labelme_to_labelstudio(dataset_root: Path, backup: bool = True) -> Dict[str, Any]:
    """Convert all LabelMe-style JSONs in the dataset to Label Studio format."""
    stats = {
        "total": 0,
        "converted": 0,
        "already_labelstudio": 0,
        "skipped": 0,
        "errors": [],
        "sample_preview": None,
    }

    ann_dirs, img_dirs = find_annotation_image_pairs(dataset_root)
    if not ann_dirs:
        stats["errors"].append(f"No annotation directories found in {dataset_root}")
        return stats

    for ann_dir, img_dir in zip(ann_dirs, img_dirs):
        for json_path in ann_dir.rglob("*.json"):
            stats["total"] += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Detect Label Studio style to avoid double-conversion
                if "labels" in data and "shapes" not in data:
                    stats["already_labelstudio"] += 1
                    continue

                if "shapes" not in data:
                    stats["skipped"] += 1
                    continue

                image_hint = data.get("imagePath") or data.get("image") or ""
                image_path = _resolve_image_path(img_dir, image_hint, json_path.stem)

                ls_payload = convert_labelme_json_to_labelstudio(data, image_path)
                if ls_payload is None:
                    stats["errors"].append(f"Could not convert (missing image/dims): {json_path}")
                    continue

                if backup:
                    backup_path = json_path.with_suffix(".json.backup")
                    if not backup_path.exists():
                        shutil.copy2(json_path, backup_path)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(ls_payload, f, indent=2, ensure_ascii=False)

                stats["converted"] += 1

                if stats["sample_preview"] is None and image_path is not None:
                    stats["sample_preview"] = {
                        "json_path": str(json_path),
                        "image_path": str(image_path),
                        "labels": ls_payload.get("labels", []),
                    }
            except Exception as exc:  # pragma: no cover - best-effort logging
                stats["errors"].append(f"{json_path}: {exc}")

    return stats
