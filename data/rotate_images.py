import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2


def _rotate_points_cw90(points: List[List[float]], orig_height: int, orig_width: int) -> List[List[float]]:
    """Rotate points 90° clockwise. New dims: (orig_width, orig_height)."""
    rotated = []
    for x, y in points:
        new_x = orig_height - y
        new_y = x
        rotated.append([new_x, new_y])
    return rotated


def _rotate_points_180(points: List[List[float]], orig_height: int, orig_width: int) -> List[List[float]]:
    """Rotate points 180° around image center (dims unchanged)."""
    rotated = []
    for x, y in points:
        new_x = orig_width - x
        new_y = orig_height - y
        # Clamp to image bounds in case of boundary floats
        new_x = min(max(new_x, 0.0), float(orig_width))
        new_y = min(max(new_y, 0.0), float(orig_height))
        rotated.append([new_x, new_y])
    return rotated


def detect_vertical_images(img_dir: Path) -> List[Tuple[Path, int, int]]:
    """Find vertical (portrait) images. Returns list of (path, height, width)."""
    vertical = []
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"]

    for ext in exts:
        for img_path in img_dir.rglob(f"*{ext}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h > w:  # Portrait
                vertical.append((img_path, h, w))

    return vertical


def rotate_image_and_update_json(
    img_path: Path, json_path: Path, dry_run: bool = False
) -> Tuple[bool, str]:
    """Rotate vertical image 90° CW and update its JSON annotations.

    Returns: (success, message)
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, f"Image read failed: {img_path}"

        h, w = img.shape[:2]
        if h <= w:
            return False, f"Not vertical (h={h}, w={w}): {img_path.name}"

        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Update image dimensions
        data["imageHeight"] = w  # New height after rotation
        data["imageWidth"] = h   # New width after rotation

        # Rotate all shapes
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if points:
                shape["points"] = _rotate_points_cw90(points, h, w)

        # For Label Studio format
        for label in data.get("labels", []):
            points = label.get("points", [])
            if points:
                # Points in Label Studio are percentages; scale based on new dimensions
                abs_points = [[p[0] / 100.0 * w, p[1] / 100.0 * h] for p in points]
                rotated_abs = _rotate_points_cw90(abs_points, h, w)
                label["points"] = [[p[0] / w * 100.0, p[1] / h * 100.0] for p in rotated_abs]

        if not dry_run:
            cv2.imwrite(str(img_path), rotated_img)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return True, f"Rotated: {img_path.name}"
    except Exception as exc:
        return False, f"{img_path.name}: {exc}"


def rotate_image_180_and_update_json(
    img_path: Path, json_path: Path, dry_run: bool = False
) -> Tuple[bool, str]:
    """Rotate image 180° and update JSON annotations accordingly."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, f"Image read failed: {img_path}"

        h, w = img.shape[:2]
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["imageHeight"] = h
        data["imageWidth"] = w

        # LabelMe shapes
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if points:
                shape["points"] = _rotate_points_180(points, h, w)

        # Label Studio percentage polygons
        for label in data.get("labels", []):
            points = label.get("points", [])
            if points:
                abs_points = [[p[0] / 100.0 * w, p[1] / 100.0 * h] for p in points]
                rotated_abs = _rotate_points_180(abs_points, h, w)
                label["points"] = [[p[0] / w * 100.0, p[1] / h * 100.0] for p in rotated_abs]

        # If this was Label Studio style, ensure downstream code sees shapes too
        if "labels" in data and "shapes" not in data:
            data["shapes"] = []
            for lbl in data.get("labels", []):
                pts = lbl.get("points", [])
                lbls = lbl.get("polygonlabels", [])
                label_name = lbls[0] if lbls else ""
                if pts:
                    abs_pts = [[p[0] / 100.0 * w, p[1] / 100.0 * h] for p in pts]
                    data["shapes"].append({"label": label_name, "points": abs_pts})

        if not dry_run:
            cv2.imwrite(str(img_path), rotated_img)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return True, f"Rotated 180°: {img_path.name}"
    except Exception as exc:  # pragma: no cover - robustness
        return False, f"{img_path.name}: {exc}"


def rotate_all_vertical_images(dataset_root: Path, ann_dirs: List[Path], img_dirs: List[Path], dry_run: bool = False) -> Dict[str, Any]:
    """Find and rotate all vertical images + update JSON annotations."""
    stats = {
        "total_vertical": 0,
        "rotated": 0,
        "errors": [],
        "sample_rotated_img": None,
        "sample_rotated_json": None,
    }

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG"]

    for img_dir in img_dirs:
        if not img_dir.exists():
            continue

        vertical_images = detect_vertical_images(img_dir)
        stats["total_vertical"] += len(vertical_images)

        for img_path, h, w in vertical_images:
            json_stem = img_path.stem
            json_path = None

            # Find corresponding JSON
            for ann_dir in ann_dirs:
                if not ann_dir.exists():
                    continue
                for ext in [".json"]:
                    candidate = next(ann_dir.rglob(f"{json_stem}{ext}"), None)
                    if candidate:
                        json_path = candidate
                        break
                if json_path:
                    break

            if not json_path:
                stats["errors"].append(f"JSON not found for: {img_path.name}")
                continue

            success, msg = rotate_image_and_update_json(img_path, json_path, dry_run=dry_run)
            if success:
                stats["rotated"] += 1
                # Store first rotated sample for preview
                if stats["sample_rotated_img"] is None:
                    stats["sample_rotated_img"] = str(img_path)
                    stats["sample_rotated_json"] = str(json_path)
            else:
                stats["errors"].append(msg)

    return stats


def rotate_all_images_180(dataset_root: Path, ann_dirs: List[Path], img_dirs: List[Path], dry_run: bool = False) -> Dict[str, Any]:
    """Rotate all images 180° across the dataset root (structure agnostic) and update JSON annotations.

    Searches all images under dataset_root (recursively) and looks for a JSON with the same stem anywhere under
    dataset_root. If found, rotates both; otherwise records an error.
    """
    stats = {
        "total_images": 0,
        "rotated": 0,
        "rotated_with_json": 0,
        "rotated_without_json": 0,
        "errors": [],
        "sample_rotated_img": None,
        "sample_rotated_json": None,
    }

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]

    dataset_root = Path(dataset_root)

    # Build stem -> json path map across entire dataset_root for fast lookup
    stem_to_json: Dict[str, Path] = {}
    for json_path in dataset_root.rglob("*.json"):
        stem_to_json[json_path.stem] = json_path

    # Iterate all images under dataset_root
    for ext in exts:
        for img_path in dataset_root.rglob(f"*{ext}"):
            if not img_path.is_file():
                continue
            stats["total_images"] += 1
            json_path = stem_to_json.get(img_path.stem)

            if not json_path:
                # Rotate image even if JSON is missing so pictures still change on disk
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["errors"].append(f"Image read failed (no JSON): {img_path.relative_to(dataset_root)}")
                    continue

                rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                if not dry_run:
                    cv2.imwrite(str(img_path), rotated_img)

                stats["rotated"] += 1
                stats["rotated_without_json"] += 1
                if stats["sample_rotated_img"] is None:
                    stats["sample_rotated_img"] = str(img_path)
                continue

            success, msg = rotate_image_180_and_update_json(img_path, json_path, dry_run=dry_run)
            if success:
                stats["rotated"] += 1
                stats["rotated_with_json"] += 1
                if stats["sample_rotated_img"] is None:
                    stats["sample_rotated_img"] = str(img_path)
                    stats["sample_rotated_json"] = str(json_path)
            else:
                stats["errors"].append(msg)

    return stats
