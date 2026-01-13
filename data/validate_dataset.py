import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from data.convert_labelstudio_to_labelme import convert_labelstudio_to_labelme

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]


def find_annotation_image_pairs(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    """Locate annotation/image directory pairs for supported layouts.

    Supports:
    - Merged: images_all / labels_all
    - Reorganized: images / labels
    - Original: <name> / <name>_Annotations
    """
    dataset_root = Path(dataset_root)
    ann_dirs: List[Path] = []
    img_dirs: List[Path] = []

    labels_all = dataset_root / "labels_all"
    images_all = dataset_root / "images_all"
    if labels_all.exists():
        ann_dirs.append(labels_all)
        img_dirs.append(images_all if images_all.exists() else dataset_root / "images")
        return ann_dirs, img_dirs

    labels_dir = dataset_root / "labels"
    images_dir = dataset_root / "images"
    if labels_dir.exists():
        ann_dirs.append(labels_dir)
        img_dirs.append(images_dir)
        return ann_dirs, img_dirs

    # Original scattered layout
    for item in dataset_root.iterdir():
        if not item.is_dir():
            continue
        if item.name.endswith("_Annotations"):
            base = item.name[:-12]
            img_dir = dataset_root / base
            if img_dir.exists():
                ann_dirs.append(item)
                img_dirs.append(img_dir)

    return ann_dirs, img_dirs


def _resolve_image_path(image_dirs: List[Path], image_hint: str, stem: str) -> Path | None:
    candidates: List[Path] = []
    hint_name = Path(image_hint).name if image_hint else ""

    for img_dir in image_dirs:
        if hint_name:
            candidates.append(img_dir / hint_name)
            candidates.extend(img_dir.rglob(hint_name))
        for ext in IMAGE_EXTENSIONS:
            candidates.append(img_dir / f"{stem}{ext}")
            candidates.extend(img_dir.rglob(f"{stem}{ext}"))

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def validate_dataset(
    dataset_root: Path,
    check_labelme: bool = True,
    check_detection: bool = False,
    check_segmentation: bool = False,
    detection_labels_root: Path | None = None,
    segmentation_masks_root: Path | None = None,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    ann_dirs, img_dirs = find_annotation_image_pairs(dataset_root)

    results: Dict[str, Any] = {
        "labelme": {"total": 0, "valid": 0, "errors": [], "warnings": []},
        "detection": {"total": 0, "valid": 0, "errors": []},
        "segmentation": {"total": 0, "valid": 0, "errors": []},
        "paths": {"images": {}, "labels": {}},
        "statistics": {
            "total_images": 0,
            "total_labels": 0,
            "images_without_labels": 0,
            "labels_without_images": 0,
            "images_without_labels_list": [],
            "labels_without_images_list": [],
            "classes": [],
            "class_counts": {},
            "total_annotations": 0,
        },
    }

    if not ann_dirs:
        results["labelme"]["errors"].append(f"Annotation klasoru bulunamadı: {dataset_root}")
        return results

    # Collect available images
    all_images: Dict[str, Path] = {}
    for img_dir in img_dirs:
        if img_dir.exists():
            for ext in IMAGE_EXTENSIONS:
                for img_path in img_dir.rglob(f"*{ext}"):
                    all_images[img_path.stem] = img_path
    results["statistics"]["total_images"] = len(all_images)
    results["paths"]["images"] = {k: str(v) for k, v in all_images.items()}

    labeled_image_stems: set[str] = set()

    if check_labelme:
        for ann_dir in ann_dirs:
            for json_path in ann_dir.rglob("*.json"):
                results["labelme"]["total"] += 1
                results["paths"]["labels"][json_path.stem] = str(json_path)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Handle both shapes (LabelMe) and labels (uniform) format
                    shapes = data.get("shapes", []) or []
                    labels = data.get("labels", []) or []
                    
                    # Check if annotation is empty (neither shapes nor labels with valid points)
                    has_annotations = False
                    if shapes:
                        for shape in shapes:
                            pts = shape.get("points", [])
                            if len(pts) >= 2:  # Support bbox (2 pts) and polygon (3+ pts)
                                has_annotations = True
                                break
                    if not has_annotations and labels:
                        for lbl in labels:
                            pts = lbl.get("points", [])
                            if len(pts) >= 2:  # Support bbox (2 pts) and polygon (3+ pts)
                                has_annotations = True
                                break
                    
                    if not has_annotations:
                        results["labelme"]["warnings"].append(f"Boş anotasyon: {json_path}")

                    image_hint = data.get("imagePath") or data.get("image") or ""
                    image_path = _resolve_image_path(img_dirs, image_hint, json_path.stem)
                    if image_path is None:
                        results["labelme"]["errors"].append(f"Görüntü bulunamadı: {json_path}")
                        continue

                    labeled_image_stems.add(image_path.stem)

                    # Process shapes (LabelMe format)
                    for shape in shapes:
                        label = shape.get("label")
                        pts = shape.get("points", [])
                        if not label or not pts or len(pts) < 2:  # Support bbox (2 pts) and polygon (3+ pts)
                            results["labelme"]["errors"].append(f"Geçersiz shape: {json_path}")
                            continue
                        results["statistics"]["total_annotations"] += 1
                        results["statistics"]["class_counts"].setdefault(label, 0)
                        results["statistics"]["class_counts"][label] += 1
                    
                    # Process labels (uniform format)
                    for lbl in labels:
                        label_list = lbl.get("polygonlabels", [])
                        label = label_list[0] if label_list else None
                        pts = lbl.get("points", [])
                        if not label or not pts or len(pts) < 2:  # Support bbox (2 pts) and polygon (3+ pts)
                            results["labelme"]["errors"].append(f"Geçersiz label: {json_path}")
                            continue
                        results["statistics"]["total_annotations"] += 1
                        results["statistics"]["class_counts"].setdefault(label, 0)
                        results["statistics"]["class_counts"][label] += 1

                    results["labelme"]["valid"] += 1
                except Exception as exc:  # pragma: no cover - robustness
                    results["labelme"]["errors"].append(f"{json_path}: {exc}")

    # Compute image/label mismatches
    all_label_stems = set()
    for ann_dir in ann_dirs:
        all_label_stems.update([p.stem for p in ann_dir.rglob("*.json")])

    images_without_labels = sorted(set(all_images.keys()) - all_label_stems)
    labels_without_images = sorted(all_label_stems - set(all_images.keys()))

    results["statistics"]["images_without_labels"] = len(images_without_labels)
    results["statistics"]["labels_without_images"] = len(labels_without_images)
    results["statistics"]["images_without_labels_list"] = images_without_labels
    results["statistics"]["labels_without_images_list"] = labels_without_images
    results["statistics"]["total_labels"] = len(all_label_stems)
    results["statistics"]["classes"] = sorted(results["statistics"]["class_counts"].keys())

    # Detection/segmentation placeholders
    if check_detection:
        results["detection"]["errors"].append("Detection kontrolü uygulanmadı (stub)")
    if check_segmentation:
        results["segmentation"]["errors"].append("Segmentation kontrolü uygulanmadı (stub)")

    return results

