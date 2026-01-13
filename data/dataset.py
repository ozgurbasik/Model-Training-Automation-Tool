from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Augmentation için
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def default_detection_transform(image: np.ndarray, boxes: np.ndarray):
    """Simple resize + normalize without external deps."""
    h, w = image.shape[:2]
    target_size = 640
    scale_y = target_size / h
    scale_x = target_size / w
    image = cv2.resize(image, (target_size, target_size))
    if boxes.size > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # simple normalization
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image), boxes


def default_segmentation_transform(image: np.ndarray, mask: np.ndarray):
    h, w = image.shape[:2]
    target_size = 640
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64))


class DetectionDataset(Dataset):
    """
    Reads images from a list file and YOLO-style txt labels,
    returns data ready for torchvision detection models.
    """

    def __init__(
        self,
        list_file: Path,
        labels_root: Path,
        dataset_root: Optional[Path] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.image_paths: List[Path] = []
        # If dataset_root is not provided, infer from list_file location (splits_dir)
        if dataset_root is None:
            dataset_root = list_file.parent
        
        self.dataset_root = Path(dataset_root).resolve()
        
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Resolve relative paths from train.txt against dataset_root
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = self.dataset_root / img_path
                self.image_paths.append(img_path)
        self.labels_root = Path(labels_root).resolve()
        self.transforms = transforms or default_detection_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _labelme_to_boxes_labels(
        self, json_path: Path, img_w: int, img_h: int, label_map: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert LabelMe JSON to bounding boxes."""
        import json
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        boxes: List[List[float]] = []
        labels: List[int] = []
        
        for shape in data.get("shapes", []):
            label = shape.get("label")
            points = shape.get("points")
            if not label or not points or len(points) < 3:
                continue
            
            class_id = label_map.get(label)
            if class_id is None:
                continue
            
            # Convert polygon to bbox
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min = max(0.0, min(x_coords))
            x_max = max(0.0, min(img_w - 1, max(x_coords)))
            y_min = max(0.0, min(y_coords))
            y_max = max(0.0, min(img_h - 1, max(y_coords)))
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id + 1)  # +1 for torchvision (0 is background)
        
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

    def _yolo_to_boxes_labels(
        self, label_path: Path, img_w: int, img_h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        boxes: List[List[float]] = []
        labels: List[int] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                cls_id = int(cls)
                cx = float(cx) * img_w
                cy = float(cy) * img_h
                bw = float(w) * img_w
                bh = float(h) * img_h
                x_min = cx - bw / 2.0
                y_min = cy - bh / 2.0
                x_max = cx + bw / 2.0
                y_max = cy + bh / 2.0
                boxes.append([x_min, y_min, x_max, y_max])
                # +1 because torchvision detection expects 1..num_classes, 0 is background
                labels.append(cls_id + 1)
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        h, w = image.shape[:2]

        # Get split name (train/val/test) from path relative to dataset_root
        # Path format: splits/train/images/... or splits/val/images/...
        try:
            rel_path = img_path.relative_to(self.dataset_root)
            split_name = rel_path.parts[0]  # First part is train/val/test
        except ValueError:
            # Fallback: try to get from parent directory name
            split_name = img_path.parent.parent.name if len(img_path.parts) >= 3 else img_path.parent.name
        # Primary: labels_root/{split}/<file>.txt
        label_path = self.labels_root / split_name / f"{img_path.stem}.txt"
        # Common alternatives (flat labels folder or nested labels/labels_all)
        if not label_path.exists():
            alt_candidates = [
                self.labels_root / f"{img_path.stem}.txt",
                self.labels_root / "labels" / f"{img_path.stem}.txt",
                self.labels_root / "labels_all" / f"{img_path.stem}.txt",
            ]
            # If we have a relative path, try to mirror the structure but swap images->labels
            try:
                rel_no_ext = rel_path.with_suffix("")
                alt_candidates.append(self.labels_root / rel_no_ext.parent / f"{rel_no_ext.name}.txt")
                # Swap images to labels in the path
                parts = list(rel_no_ext.parts)
                parts_swapped = ["labels" if p == "images" else p for p in parts]
                alt_candidates.append(self.labels_root / Path(*parts_swapped).with_suffix(".txt"))
            except Exception:
                pass
            for cand in alt_candidates:
                if cand.exists():
                    label_path = cand
                    break
        
        # If YOLO .txt doesn't exist, try to find LabelMe JSON and convert on-the-fly
        if not label_path.exists():
            # Try to find LabelMe JSON in various locations
            json_paths = [
                self.labels_root / split_name / f"{img_path.stem}.json",
                img_path.parent.parent / "labels" / f"{img_path.stem}.json",  # splits/train/labels/
                img_path.parent.parent.parent / "labels" / f"{img_path.stem}.json",  # Dataset_Merged/labels/
            ]
            
            json_path = None
            for jp in json_paths:
                if jp.exists():
                    json_path = jp
                    break
            
            if json_path:
                # Load label map
                classes_file = self.labels_root / "classes.txt"
                if not classes_file.exists():
                    classes_file = self.labels_root.parent / "classes.txt"
                label_map = {}
                if classes_file.exists():
                    with open(classes_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(None, 1)
                            if len(parts) == 2:
                                label_map[parts[1]] = int(parts[0])
                
                # Convert LabelMe to boxes
                boxes, labels = self._labelme_to_boxes_labels(json_path, w, h, label_map)
            else:
                boxes, labels = self._yolo_to_boxes_labels(label_path, w, h)
        else:
            boxes, labels = self._yolo_to_boxes_labels(label_path, w, h)
        image_t, boxes = self.transforms(image, boxes)

        target: Dict[str, torch.Tensor] = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        return image_t, target


class SegmentationDataset(Dataset):
    """
    Reads images from a list file and mask images from a parallel root,
    returns tensors suitable for semantic segmentation.
    """

    def __init__(
        self,
        list_file: Path,
        masks_root: Path,
        dataset_root: Optional[Path] = None,
        transforms: Optional[Callable] = None,
        split_name: Optional[str] = None,
    ) -> None:
        self.image_paths: List[Path] = []
        # If dataset_root is not provided, infer from list_file location (splits_dir)
        if dataset_root is None:
            dataset_root = list_file.parent
        
        self.dataset_root = Path(dataset_root).resolve()
        self.split_name = split_name  # Store split name if provided
        
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Resolve relative paths from train.txt against dataset_root
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = self.dataset_root / img_path
                self.image_paths.append(img_path)
        self.masks_root = Path(masks_root).resolve()
        self.transforms = transforms or default_segmentation_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        
        # Use provided split_name if available
        if self.split_name:
            split_name = self.split_name
        else:
            # Get split name (train/val/test) from path relative to dataset_root
            # Path format: images_all/H1_frame_089.png or train/images/H1_frame_089.png
            try:
                rel_path = img_path.relative_to(self.dataset_root)
                split_name = rel_path.parts[0]  # First part is train/val/test or images_all
            except ValueError:
                # Fallback: try to get from parent directory name
                split_name = img_path.parent.parent.name if len(img_path.parts) >= 3 else img_path.parent.name
        
        # Try multiple mask locations (PNG first, then JSON)
        # Most common: splits/train/labels/...
        mask_path = self.masks_root / split_name / "labels" / f"{img_path.stem}.png"
        if not mask_path.exists():
            # Try JSON format (for LabelMe)
            mask_path = self.masks_root / split_name / "labels" / f"{img_path.stem}.json"
        if not mask_path.exists():
            # Alternative: splits/train/masks/...
            mask_path = self.masks_root / split_name / "masks" / f"{img_path.stem}.png"
        if not mask_path.exists():
            # Alternative: splits/train/masks/ with JSON
            mask_path = self.masks_root / split_name / "masks" / f"{img_path.stem}.json"
        if not mask_path.exists():
            # Third: masks_root/split_name/...
            mask_path = self.masks_root / split_name / f"{img_path.stem}.png"
        if not mask_path.exists():
            # Third: masks_root/split_name/... JSON
            mask_path = self.masks_root / split_name / f"{img_path.stem}.json"
        
        if not mask_path.exists():
            # Create empty mask (all zeros) for images without masks
            print(f"⚠️ Using empty mask for {img_path.name}: no mask found at {self.masks_root / split_name}")
            # Create empty mask with same size as image
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            # Load mask - handle JSON (LabelMe) or PNG
            if mask_path.suffix.lower() == ".json":
                # Load LabelMe JSON and convert to mask
                from data.labelme_to_segmentation import create_mask_for_annotation
                import json
                try:
                    with open(mask_path, 'r') as f:
                        anno = json.load(f)
                    h, w = image.shape[:2]
                    mask = create_mask_for_annotation(anno, h, w)
                except Exception as e:
                    print(f"⚠️ Failed to load mask from JSON {mask_path}: {e}")
                    h, w = image.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
            else:
                # Load PNG mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        image_t, mask_t = self.transforms(image, mask)
        return image_t, mask_t


def collate_detection(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


class AugmentationFactory:
    """Augmentation factory for detection and segmentation tasks."""
    
    @staticmethod
    def get_detection_transforms(
        augment: bool = False,
        img_size: int = 640,
        horizontal_flip: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        blur: float = 0.1,
        noise: float = 0.1,
        rotation: float = 10.0,
        scale: float = 0.1,
        translation: float = 0.1,
        crop_scale: float = 0.8,
        mixup: float = 0.0,
        cutmix: float = 0.0,
        mosaic: float = 0.0,
    ):
        """Get detection transforms with optional augmentation."""
        
        if not ALBUMENTATIONS_AVAILABLE:
            print("Warning: Albumentations not available, using default transforms")
            return lambda image, boxes: default_detection_transform(image, boxes)
        
        # Base transforms (always applied)
        base_transforms = [
            A.Resize(img_size, img_size),
        ]
        
        # Augmentation transforms (only if augment=True)
        augment_transforms = []
        if augment:
            if horizontal_flip > 0:
                augment_transforms.append(A.HorizontalFlip(p=horizontal_flip))
            
            if rotation > 0 or scale > 0 or translation > 0:
                augment_transforms.append(
                    A.ShiftScaleRotate(
                        shift_limit=translation,
                        scale_limit=scale,
                        rotate_limit=rotation,
                        p=0.5
                    )
                )
            
            if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
                augment_transforms.append(
                    A.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                        p=0.5
                    )
                )
            
            if blur > 0:
                augment_transforms.append(A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=blur))
            
            if noise > 0:
                augment_transforms.append(A.OneOf([
                    A.GaussNoise(),
                    A.ISONoise(),
                ], p=noise))
            
            if crop_scale > 0:
                augment_transforms.append(
                    A.RandomResizedCrop(
                        size=(img_size, img_size),
                        scale=(crop_scale, 1.0),
                        ratio=(0.8, 1.2),
                        p=0.3
                    )
                )
        
        # Normalization and tensor conversion
        final_transforms = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
        
        # Combine all transforms
        all_transforms = base_transforms + augment_transforms + final_transforms
        
        # Create albumentations transform
        transform = A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3,
                clip=True
            )
        )
        
        def apply_transform(image, boxes):
            """Apply transform to image and boxes."""
            if len(boxes) == 0:
                # No boxes, still provide empty bboxes/labels to satisfy label_fields
                transformed = transform(image=image, bboxes=[], labels=[])
                image_t = transformed['image']
                return image_t, np.zeros((0, 4), dtype=np.float32)
            
            # Convert boxes to albumentations format
            bboxes = boxes.tolist()
            labels = list(range(len(bboxes)))  # Dummy labels for albumentations
            
            # Apply transform
            transformed = transform(
                image=image,
                bboxes=bboxes,
                labels=labels
            )
            
            image_t = transformed['image']
            transformed_bboxes = transformed['bboxes']
            
            # Convert back to numpy array
            if transformed_bboxes:
                boxes_t = np.array(transformed_bboxes, dtype=np.float32)
            else:
                boxes_t = np.zeros((0, 4), dtype=np.float32)
            
            return image_t, boxes_t
        
        return apply_transform
    
    @staticmethod
    def get_segmentation_transforms(
        augment: bool = False,
        img_size: int = 640,
        horizontal_flip: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        blur: float = 0.1,
        noise: float = 0.1,
        rotation: float = 10.0,
        scale: float = 0.1,
        translation: float = 0.1,
        elastic: float = 0.1,
        grid_distortion: float = 0.1,
    ):
        """Get segmentation transforms with optional augmentation."""
        
        if not ALBUMENTATIONS_AVAILABLE:
            print("Warning: Albumentations not available, using default transforms")
            return lambda image, mask: default_segmentation_transform(image, mask)
        
        # Base transforms (always applied)
        base_transforms = [
            A.Resize(img_size, img_size),
        ]
        
        # Augmentation transforms (only if augment=True)
        augment_transforms = []
        if augment:
            if horizontal_flip > 0:
                augment_transforms.append(A.HorizontalFlip(p=horizontal_flip))
            
            if rotation > 0 or scale > 0 or translation > 0:
                augment_transforms.append(
                    A.ShiftScaleRotate(
                        shift_limit=translation,
                        scale_limit=scale,
                        rotate_limit=rotation,
                        p=0.5
                    )
                )
            
            if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
                augment_transforms.append(
                    A.ColorJitter(
                        brightness_limit=brightness,
                        contrast_limit=contrast,
                        saturation_limit=saturation,
                        hue_limit=hue,
                        p=0.5
                    )
                )
            
            if blur > 0:
                augment_transforms.append(A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=blur))
            
            if noise > 0:
                augment_transforms.append(A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(),
                ], p=noise))
            
            if elastic > 0:
                augment_transforms.append(A.ElasticTransform(p=elastic))
            
            if grid_distortion > 0:
                augment_transforms.append(A.GridDistortion(p=grid_distortion))
        
        # Normalization and tensor conversion
        final_transforms = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
        
        # Combine all transforms
        all_transforms = base_transforms + augment_transforms + final_transforms
        
        # Create albumentations transform
        transform = A.Compose(all_transforms)
        
        def apply_transform(image, mask):
            """Apply transform to image and mask."""
            transformed = transform(image=image, mask=mask)
            image_t = transformed['image']
            mask_t = transformed['mask']
            return image_t, mask_t
        
        return apply_transform


