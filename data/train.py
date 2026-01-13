import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Mixed precision training için
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

# Mixed precision training için
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    fcn_resnet50,
)
from torchvision.ops import box_iou

from data.dataset import DetectionDataset, SegmentationDataset, collate_detection, AugmentationFactory
from data.labelme_to_detection import process_annotation, collect_label_map, polygon_to_bbox
from data.labelme_to_segmentation import create_mask_for_annotation, collect_label_map as collect_seg_label_map
from data.evaluate_segmentation import evaluate_segmentation

import mlflow
import mlflow.pytorch
import cv2
import json
from PIL import Image, ImageDraw
import numpy as np

# YOLO modelleri için
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Segmentation modelleri için
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# Timm için
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Transformers için (DETR, Mask2Former)
try:
    from transformers import AutoModelForObjectDetection, AutoModelForImageSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Detectron2 için (DETR, Mask2Former, Cascade R-CNN)
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

# MMDetection için (FCOS, ATSS, EfficientDet)
try:
    from mmdet import apis
    from mmdet.models import build_detector
    from mmdet.apis import init_detector
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False

# MMSegmentation için (SegNeXt, DDRNet, PIDNet, TopFormer)
try:
    from mmseg.models import build_segmentor
    from mmseg.apis import init_segmentor
    MMSEG_AVAILABLE = True
except ImportError:
    MMSEG_AVAILABLE = False


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Reverse simple normalization ((x-0.5)/0.5) back to uint8 RGB."""
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = ((img * 0.5) + 0.5).clip(0, 1)
    return (img * 255).astype(np.uint8)


def _build_palette(num_classes: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
        ],
        dtype=np.uint8,
    )
    if num_classes <= len(base):
        return base[:num_classes]
    reps = int(np.ceil(num_classes / len(base)))
    return np.tile(base, (reps, 1))[:num_classes]


def create_detection_model(model_name: str, num_classes: int):
    """Create detection model based on model_name."""
    # YOLO models (return special marker - handled separately)
    if model_name.startswith("yolov8") or model_name.startswith("yolov5"):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package required for YOLO models. Install: pip install ultralytics")
        return {"type": "yolo", "name": model_name}
    
    if model_name.startswith("yolov7") or model_name.startswith("yolov6") or model_name.startswith("yolox"):
        raise NotImplementedError(f"{model_name} requires custom implementation. Use YOLOv8/YOLOv5 for now.")
    
    if model_name.startswith("rt_detr"):
        raise NotImplementedError("RT-DETR requires custom implementation.")
    
    # DETR (Detection Transformer)
    if model_name == "detr":
        if TRANSFORMERS_AVAILABLE:
            try:
                model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                # Update num_classes
                model.config.num_labels = num_classes
                return model
            except:
                pass
        # Fallback to torchvision DETR if available
        try:
            from torchvision.models.detection import detr_resnet50
            model = detr_resnet50(pretrained=True, num_classes=num_classes)
            return model
        except:
            raise ImportError("DETR requires transformers or torchvision>=0.13. Install: pip install transformers")
    
    # Cascade R-CNN
    if model_name == "cascade_rcnn":
        try:
            from torchvision.models.detection import cascade_rcnn_resnet50_fpn
            model = cascade_rcnn_resnet50_fpn(weights="DEFAULT")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
            model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)
            return model
        except:
            if DETECTRON2_AVAILABLE:
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/cascade_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
                model = build_model(cfg)
                return model
            raise ImportError("Cascade R-CNN requires torchvision>=0.13 or detectron2")
    
    # EfficientDet
    if model_name.startswith("efficientdet"):
        if MMDET_AVAILABLE:
            # Will be initialized with config file
            return {"type": "mmdet", "name": model_name}
        # Fallback: try timm
        if TIMM_AVAILABLE:
            try:
                model = timm.create_model("efficientdet_d0", pretrained=True, num_classes=num_classes)
                return model
            except:
                pass
        raise ImportError("EfficientDet requires mmdetection or timm. Install: pip install mmdetection")
    
    # FCOS
    if model_name == "fcos":
        if MMDET_AVAILABLE:
            return {"type": "mmdet", "name": model_name}
        raise ImportError("FCOS requires mmdetection. Install: pip install mmdetection")
    
    # ATSS
    if model_name == "atss":
        if MMDET_AVAILABLE:
            return {"type": "mmdet", "name": model_name}
        raise ImportError("ATSS requires mmdetection. Install: pip install mmdetection")
    
    # Torchvision models
    if model_name == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    elif model_name == "retinanet_resnet50_fpn":
        model = retinanet_resnet50_fpn(weights="DEFAULT")
    elif model_name == "ssd300_vgg16":
        model = ssd300_vgg16(weights="DEFAULT")
    elif model_name == "centernet":
        raise NotImplementedError("CenterNet requires custom implementation.")
    else:
        raise ValueError(f"Unsupported detection model_name: {model_name}")

    # Update classifier head for torchvision models
    if hasattr(model, "roi_heads"):
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, num_classes)
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, num_classes * 4)
    elif hasattr(model, "head"):  # SSD
        # SSD head modification would go here
        pass
    
    return model


def create_segmentation_model(model_name: str, num_classes: int):
    """Create segmentation model based on model_name."""
    # Mask2Former
    if model_name == "mask2former":
        if TRANSFORMERS_AVAILABLE:
            try:
                model = AutoModelForImageSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic")
                # Update num_classes
                model.config.num_labels = num_classes
                return model
            except:
                pass
        if DETECTRON2_AVAILABLE:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/maskformer_R50_bs16_50ep.yaml"))
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
            model = build_model(cfg)
            return model
        raise ImportError("Mask2Former requires transformers or detectron2. Install: pip install transformers detectron2")
    
    # SegNeXt
    if model_name == "segnext":
        if MMSEG_AVAILABLE:
            return {"type": "mmseg", "name": model_name}
        raise ImportError("SegNeXt requires mmsegmentation. Install: pip install mmsegmentation")
    
    # BiSeNetV2
    if model_name == "bisenetv2":
        if MMSEG_AVAILABLE:
            return {"type": "mmseg", "name": model_name}
        # Try SMP fallback
        if SMP_AVAILABLE:
            # BiSeNetV2-like architecture via SMP
            model = smp.Unet(encoder_name="timm-efficientnet-b0", encoder_weights="imagenet", classes=num_classes)
            return model
        raise ImportError("BiSeNetV2 requires mmsegmentation or segmentation-models-pytorch")
    
    # DDRNet
    if model_name == "ddrnet":
        if MMSEG_AVAILABLE:
            return {"type": "mmseg", "name": model_name}
        raise ImportError("DDRNet requires mmsegmentation. Install: pip install mmsegmentation")
    
    # PIDNet
    if model_name == "pidnet":
        if MMSEG_AVAILABLE:
            return {"type": "mmseg", "name": model_name}
        raise ImportError("PIDNet requires mmsegmentation. Install: pip install mmsegmentation")
    
    # TopFormer
    if model_name == "topformer":
        if MMSEG_AVAILABLE:
            return {"type": "mmseg", "name": model_name}
        raise ImportError("TopFormer requires mmsegmentation. Install: pip install mmsegmentation")
    
    # Segmentation Models PyTorch models
    if SMP_AVAILABLE:
        if model_name == "unet":
            model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes)
            return model
        elif model_name == "unet++":
            model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes)
            return model
        elif model_name == "deeplabv3+":
            model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", classes=num_classes)
            return model
        elif model_name == "pspnet":
            model = smp.PSPNet(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes)
            return model
        elif model_name == "fpn":
            model = smp.FPN(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes)
            return model
        elif model_name == "linknet":
            model = smp.Linknet(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes)
            return model
        elif model_name == "manet":
            model = smp.MAnet(encoder_name="resnet34", encoder_weights="imagenet", classes=num_classes)
            return model
    
    # Timm models
    if TIMM_AVAILABLE:
        if model_name == "segformer":
            # SegFormer via timm
            model = timm.create_model("segformer_b0", pretrained=True, num_classes=num_classes)
            return model
        elif model_name == "hrnet":
            model = timm.create_model("hrnet_w18", pretrained=True, num_classes=num_classes)
            return model
    
    # Torchvision models
    if model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights="DEFAULT")
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        return model
    elif model_name == "fcn_resnet50":
        model = fcn_resnet50(weights="DEFAULT")
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        return model
    
    # Custom implementations
    if model_name == "fast_scnn":
        raise NotImplementedError("Fast-SCNN requires custom implementation.")
    elif model_name == "bisenet":
        raise NotImplementedError("BiSeNet requires custom implementation.")
    elif model_name == "enet":
        raise NotImplementedError("ENet requires custom implementation.")
    
    raise ValueError(f"Unsupported segmentation model_name: {model_name}")


def train_yolo_detection(cfg: Dict, config_path: Path):
    """Train YOLO models using ultralytics API."""
    data_cfg = cfg["data"]
    common = cfg["common"]
    det_cfg = cfg.get("detection", {})
    
    # Get run name for directory structure
    run_name = cfg.get("run_name", "default_run")
    safe_run_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_name))
    
    model_name = det_cfg.get("model_name", "yolov8n")
    # Map model names to ultralytics format
    if model_name.startswith("yolov5"):
        yolo_size = model_name.replace("yolov5", "") or "n"
        yolo_model_name = f"yolov5{yolo_size}.pt"
    elif model_name.startswith("yolov8"):
        yolo_size = model_name.replace("yolov8", "") or "n"
        yolo_model_name = f"yolov8{yolo_size}.pt"
    else:
        yolo_model_name = "yolov8n.pt"  # default
    
    # Create YOLO model
    model = YOLO(yolo_model_name)
    
    # Prepare data config for YOLO (needs YAML format)
    splits_dir = Path(data_cfg["splits_dir"]).resolve()
    dataset_root = Path(data_cfg.get("dataset_root", "")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path(__file__).resolve().parent.parent / dataset_root).resolve()

    # Heuristic: pick the folder that actually contains images_all
    candidates = [
        dataset_root,
        dataset_root / "RcCArDataset",
        Path(__file__).resolve().parent.parent / "DataSet" / "RcCArDataset",
    ]
    for cand in candidates:
        if (cand / "images_all").exists():
            dataset_root = cand.resolve()
            break
    else:
        # Fallback: if original exists use it, else leave as-is
        if dataset_root.exists():
            dataset_root = dataset_root.resolve()

    # Ensure YOLO-friendly aliases exist: images -> images_all, labels -> labels_all
    import os
    import subprocess
    def _ensure_junction(alias: Path, target: Path):
        try:
            if target.exists() and not alias.exists():
                if os.name == 'nt':
                    subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(target)], check=True)
                else:
                    os.symlink(str(target), str(alias), target_is_directory=True)
        except Exception:
            # Non-fatal: fallback to direct paths if junctions cannot be created
            pass

    _ensure_junction(dataset_root / 'images', dataset_root / 'images_all')
    _ensure_junction(dataset_root / 'labels', dataset_root / 'labels_all')

    # Labels root: default to sibling labels_all under dataset_root
    labels_root_cfg = data_cfg.get("detection_labels_root", "")
    labels_root = Path(labels_root_cfg).expanduser()
    if not labels_root.is_absolute():
        labels_root = (dataset_root / labels_root_cfg).resolve()
    if not labels_root.exists() and (dataset_root / "labels_all").exists():
        labels_root = (dataset_root / "labels_all").resolve()
    if not labels_root.exists():
        labels_root = labels_root.resolve()

    # Load class names from classes.txt if available
    class_names: list[str] = []
    class_files = [
        labels_root / "classes.txt",
        dataset_root / "labels" / "classes.txt",
        dataset_root / "labels_all" / "classes.txt",
    ]
    for cls_path in class_files:
        if cls_path.exists():
            with open(cls_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) == 1:
                        class_names.append(parts[0])
                    else:
                        class_names.append(" ".join(parts[1:]))
            break
    if not class_names:
        class_names = ["object"]
    import tempfile

    # Rewrite train/val file lists to absolute paths so ultralytics can find images/labels
    def _rewrite_list(list_path: Path) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        with open(list_path, "r", encoding="utf-8") as src, tmp:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                # Map images_all paths to images alias for YOLO to derive labels
                if line.startswith('images_all/'):
                    mapped = 'images/' + line[len('images_all/'):]  # images/<filename>
                else:
                    mapped = line
                # Write absolute path WITHOUT resolving symlinks/junctions to preserve 'images' in string
                abs_path = (dataset_root / mapped)
                tmp.write(str(abs_path) + "\n")
        return Path(tmp.name)

    # Convert LabelMe labels to YOLO format BEFORE training
    print("\n🔄 Converting labels to YOLO format...")
    convert_labels_for_detection(splits_dir, labels_root, dataset_root)
    print("✅ Label conversion complete\n")
    
    train_list = _rewrite_list(splits_dir / "train.txt")
    val_list = _rewrite_list(splits_dir / "val.txt")
    
    # YOLO expects a data.yaml file
    import os
    temp_data_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_data_yaml.write(
        f"""
train: {train_list}
val: {val_list}
nc: {len(class_names)}
names: {class_names}
        """
    )
    temp_data_yaml.close()
    
    # Create run-specific output directory
    base_output_dir = Path(common["output_dir"])
    output_dir = base_output_dir / safe_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Model kayıt dizini: {output_dir}")
    
    # Train with ultralytics (YOLO creates its own structure: project/name/weights/best.pt)
    results = model.train(
        data=temp_data_yaml.name,
        epochs=common["epochs"],
        imgsz=common["img_size"],
        batch=common["batch_size"],
        lr0=common["lr"],
        weight_decay=common["weight_decay"],
        project=str(base_output_dir),
        name=safe_run_name,
    )

    # Determine actual YOLO save directory (handles auto-incremented names)
    try:
        yolo_save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    except Exception:
        yolo_save_dir = None
    effective_output_dir = yolo_save_dir if (yolo_save_dir and yolo_save_dir.exists()) else output_dir
    
    # Log metrics to MLflow
    def _sanitize_mlflow_name(name: str):
        """Sanitize MLflow metric names to allowed characters.
        Allowed: alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), slashes (/).
        Any other character will be replaced by an underscore.
        """
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. /")
        return "".join((c if c in allowed_chars else "_") for c in name)

    if hasattr(results, "results_dict"):
        for key, value in results.results_dict.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            metric_name = _sanitize_mlflow_name(f"yolo_{key}")
            mlflow.log_metric(metric_name, numeric_value)
    
    # Log YOLO artifacts (plots, confusion matrix, etc.)
    print("📊 MLflow'a grafikler kaydediliyor...")
    artifacts_to_log = [
        "results.png",              # Training results plot
        "confusion_matrix.png",     # Confusion matrix
        "confusion_matrix_normalized.png",  # Normalized confusion matrix
        "labels.jpg",               # Label distribution
        "labels_correlogram.jpg",   # Label correlogram
        "results.csv",              # Results as CSV
        # Curve plots
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        # Batch visualizations
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_labels.jpg",
        "val_batch2_pred.jpg",
    ]
    
    for artifact_name in artifacts_to_log:
        artifact_path = effective_output_dir / artifact_name
        if artifact_path.exists():
            try:
                mlflow.log_artifact(str(artifact_path))
                print(f"   ✅ {artifact_name} kaydedildi")
            except Exception as e:
                print(f"   ⚠️ {artifact_name} kaydedilemedi: {e}")
    
    # YOLO saves to base_output_dir/safe_run_name/weights/best.pt
    yolo_model_path = effective_output_dir / "weights" / "best.pt"
    if yolo_model_path.exists():
        # Also copy to main output directory for consistency
        final_model_path = effective_output_dir / "yolo_best.pt"
        import shutil
        shutil.copy2(yolo_model_path, final_model_path)
        mlflow.log_artifact(str(final_model_path))
        print(f"   ✅ YOLO model kaydedildi: {final_model_path}")
        print(f"   📁 YOLO output: {yolo_model_path}")
        
        # Log example predictions on a few validation images
        print("\n📸 Örnek tahminler kaydediliyor...")
        try:
            from ultralytics import YOLO as UltraYOLO
            best_model = UltraYOLO(str(yolo_model_path))
            
            # Create predictions dir
            predictions_dir = effective_output_dir / "example_predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            
            # Load a few validation images and get predictions
            val_image_list = []
            if Path(val_list).exists():
                with open(val_list, 'r') as f:
                    val_image_list = [line.strip() for line in f if line.strip()][:5]  # First 5 images
            
            for idx, img_rel_path in enumerate(val_image_list):
                try:
                    # val_list may contain absolute paths; handle both cases
                    p = Path(img_rel_path)
                    img_path = p if p.is_absolute() else (dataset_root / img_rel_path)
                    if not img_path.exists():
                        continue
                    
                    # Predict
                    results = best_model.predict(str(img_path), conf=0.5, verbose=False)
                    
                    # Save prediction
                    pred_save_path = predictions_dir / f"prediction_{idx+1}.jpg"
                    if len(results) > 0 and results[0].plot() is not None:
                        pred_img = results[0].plot()
                        from PIL import Image as PILImage
                        PILImage.fromarray(pred_img).save(pred_save_path)
                        mlflow.log_artifact(str(pred_save_path))
                        print(f"   ✅ Tahmin {idx+1} kaydedildi")
                except Exception as e:
                    print(f"   ⚠️ Tahmin {idx+1} işlenemedi: {e}")
        except Exception as e:
            print(f"   ⚠️ Örnek tahminler kaydedilemedi: {e}")
    else:
        print(f"   ⚠️ YOLO model bulunamadı: {yolo_model_path}")
    
    os.unlink(temp_data_yaml.name)
    os.unlink(train_list)
    os.unlink(val_list)


def convert_labels_for_detection(splits_dir: Path, labels_root: Path, dataset_root: Path):
    """Convert LabelMe JSON to YOLO format if needed before training."""
    print("🔍 Checking detection labels format...")
    
    # Read all image paths from train.txt and val.txt
    all_image_paths = []
    for split_file in [splits_dir / "train.txt", splits_dir / "val.txt"]:
        if not split_file.exists():
            continue
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path = dataset_root / line
                    if img_path.exists():
                        all_image_paths.append(img_path)
    
    if not all_image_paths:
        print("⚠️ No images found in train.txt/val.txt")
        return
    
    # Check which images need conversion
    # YOLO expects labels in same structure as images but with 'labels' instead of 'images'
    # E.g., images_all/xxx.jpg -> labels_all/xxx.txt OR images/xxx.jpg -> labels/xxx.txt
    needs_conversion = []
    for img_path in all_image_paths:
        # Derive label path: replace images_all with labels_all (or images with labels)
        rel_path = img_path.relative_to(dataset_root)
        rel_str = str(rel_path).replace("\\", "/")
        
        if rel_str.startswith("images_all/"):
            label_rel = rel_str.replace("images_all/", "labels_all/", 1)
        elif rel_str.startswith("images/"):
            label_rel = rel_str.replace("images/", "labels/", 1)
        else:
            # Fallback: just put in labels_all
            label_rel = f"labels_all/{Path(rel_str).name}"
        
        label_path = dataset_root / label_rel
        label_path = label_path.with_suffix(".txt")
        
        if not label_path.exists():
            needs_conversion.append((img_path, label_path))
    
    if not needs_conversion:
        print("✅ All detection labels are in YOLO format")
        return
    
    print(f"🔄 Converting {len(needs_conversion)} LabelMe JSON files to YOLO format...")
    
    # Find LabelMe JSON files in labels_all directory
    json_files = {}
    labels_all_dir = dataset_root / "labels_all"
    
    if not labels_all_dir.exists():
        print(f"⚠️ Labels directory not found: {labels_all_dir}")
        return
    
    for img_path, target_label_path in needs_conversion:
        # Look for JSON file in labels_all with same stem as image
        json_path = labels_all_dir / f"{img_path.stem}.json"
        if json_path.exists():
            json_files[img_path] = (json_path, target_label_path)
        else:
            print(f"   ⚠️ No JSON found for: {img_path.name}")
    
    if not json_files:
        print("⚠️ No LabelMe JSON files found for conversion")
        return
    
    # Collect label map from all JSON files
    label_map = collect_label_map([labels_all_dir])
    if not label_map:
        print("⚠️ No labels found in JSON files")
        return
    
    print(f"📋 Found {len(label_map)} classes: {list(label_map.keys())}")
    
    # Convert each JSON to YOLO format
    converted = 0
    empty_labels = 0
    for img_path, (json_path, target_label_path) in json_files.items():
        # Ensure output directory exists
        target_label_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert JSON to YOLO format - process_annotation writes to labels_dir/{stem}.txt
        # so we pass the parent and then verify the output
        try:
            process_annotation(json_path, img_path.parent, target_label_path.parent, label_map)
            # Verify the file was created
            expected_output = target_label_path.parent / f"{json_path.stem}.txt"
            if expected_output.exists():
                # If filenames differ, move it to the correct location
                if expected_output != target_label_path:
                    expected_output.rename(target_label_path)
                converted += 1
                if converted <= 3:  # Show first few for debugging
                    print(f"   ✅ {img_path.name} -> {target_label_path.name}")
            else:
                # If JSON has no annotations, write an empty label file so training treats it as background-only
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    shapes = data.get("shapes") or []
                    labels_list = data.get("labels") or []
                    if not shapes and not labels_list:
                        target_label_path.touch()
                        empty_labels += 1
                        converted += 1
                        if converted <= 3:
                            print(f"   ℹ️ {img_path.name} has no annotations; wrote empty label file")
                    else:
                        print(f"   ⚠️ No output generated for {img_path.name}")
                except Exception as parse_err:
                    print(f"   ⚠️ No output generated for {img_path.name} (parse error: {parse_err})")
        except Exception as e:
            print(f"   ⚠️ Failed to convert {img_path.name}: {e}")
    
    # Save classes.txt in labels_all directory
    classes_file = labels_all_dir / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for name, cid in sorted(label_map.items(), key=lambda x: x[1]):
            f.write(f"{cid} {name}\n")
    
    print(f"✅ Converted {converted}/{len(needs_conversion)} files to YOLO format (empty labels: {empty_labels})")


def convert_masks_for_segmentation(splits_dir: Path, masks_root: Path, dataset_root: Path):
    """Convert LabelMe JSON to segmentation masks if needed before training."""
    print("🔍 Checking segmentation masks format...")
    
    # First, collect label map from ALL splits to ensure consistent class ordering
    all_label_dirs = []
    for split_name in ["train", "val"]:
        labels_dir = splits_dir / split_name / "labels"
        if labels_dir.exists():
            all_label_dirs.append(labels_dir)
    
    if not all_label_dirs:
        print("⚠️ No label directories found in any split")
        return
    
    # Collect label map ONCE from all splits - ensures consistent ordering
    global_label_map = collect_seg_label_map(all_label_dirs)
    if not global_label_map:
        print("⚠️ No labels found in JSON files")
        return
    
    print(f"📋 Found {len(global_label_map)} classes (global): {list(global_label_map.keys())}")
    
    # Now process each split using the SAME global label map
    for split_name in ["train", "val"]:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            continue
        
        # Read image paths
        with open(split_file, "r", encoding="utf-8") as f:
            image_lines = [line.strip() for line in f if line.strip()]
        
        if not image_lines:
            continue
        
        print(f"\n📁 Processing {split_name} split ({len(image_lines)} images)...")
        
        # Check for JSON labels in splits/{split_name}/labels/
        labels_dir = splits_dir / split_name / "labels"
        if not labels_dir.exists():
            print(f"⚠️ Labels directory not found: {labels_dir}")
            continue
        
        json_files = sorted(labels_dir.glob("*.json"))  # Sort for consistency
        if not json_files:
            print(f"⚠️ No JSON files found in {labels_dir}")
            continue
        
        print(f"📋 Processing {len(json_files)} JSON files with global label map")
        
        # Output directory for masks (same as labels dir - we'll put PNG alongside JSON)
        output_dir = labels_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get images directory
        images_dir = splits_dir / split_name / "images"
        if not images_dir.exists():
            print(f"⚠️ Images directory not found: {images_dir}")
            continue
        
        # Convert each JSON to mask using the GLOBAL label map
        converted = 0
        skipped = 0
        for json_path in json_files:
            mask_png = output_dir / f"{json_path.stem}.png"
            if mask_png.exists():
                skipped += 1
                continue  # Skip if mask already exists
            
            # Use create_mask_for_annotation function with global label map
            try:
                create_mask_for_annotation(json_path, images_dir, output_dir, global_label_map)
                # Check if mask was actually created
                if mask_png.exists():
                    converted += 1
                else:
                    print(f"⚠️ Mask not created for {json_path.name}")
            except Exception as e:
                print(f"❌ Error converting {json_path.name}: {e}")
        
        print(f"✅ Converted: {converted}, Skipped (already exists): {skipped}, Total JSON: {len(json_files)}")
        
        # Save classes.txt in this split's labels directory (using global label map)
        classes_file = output_dir / "classes.txt"
        with open(classes_file, "w", encoding="utf-8") as f:
            f.write("0 background\n")
            for name, cid in sorted(global_label_map.items(), key=lambda x: x[1]):
                f.write(f"{cid} {name}\n")
        
        print(f"📝 Saved class mapping to {classes_file}")


def train_detection(cfg: Dict, config_path: Path):
    print("=" * 80)
    print("🚀 DETECTION TRAINING BAŞLADI")
    print("=" * 80)
    
    data_cfg = cfg["data"]
    common = cfg["common"]
    det_cfg = cfg.get("detection", {})

    model_name = det_cfg.get("model_name", "fasterrcnn_resnet50_fpn")
    print(f"📋 Model: {model_name}")
    print(f"📋 Epochs: {common.get('epochs', 'N/A')}")
    print(f"📋 Batch size: {common.get('batch_size', 'N/A')}")
    print(f"📋 Learning rate: {common.get('lr', 'N/A')}")
    
    # Check if YOLO model
    if model_name.startswith("yolov8") or model_name.startswith("yolov5"):
        print("🔍 YOLO modeli tespit edildi, YOLO training'e yönlendiriliyor...")
        train_yolo_detection(cfg, config_path)
        return
    
    # Check if MMDetection model
    if isinstance(model_name, dict) and model_name.get("type") == "mmdet":
        raise NotImplementedError("MMDetection models require custom training loop. Coming soon.")
    
    print("\n📂 Dataset yapılandırması:")
    splits_dir = Path(data_cfg["splits_dir"]).resolve()
    train_list = splits_dir / "train.txt"
    val_list = splits_dir / "val.txt"
    print(f"   Splits dizini: {splits_dir}")
    print(f"   Train list: {train_list} ({'✅ Var' if train_list.exists() else '❌ Yok'})")
    print(f"   Val list: {val_list} ({'✅ Var' if val_list.exists() else '❌ Yok'})")

    labels_root_cfg = Path(data_cfg["detection_labels_root"])
    # Allow pointing directly to classes.txt; treat its parent as labels_root
    if labels_root_cfg.suffix:
        labels_root = labels_root_cfg.parent.resolve()
    else:
        labels_root = labels_root_cfg.resolve()
    print(f"   Labels root: {labels_root}")
    
    # Prefer explicit dataset_root from config; otherwise fallback to splits_dir
    dataset_root_cfg = Path(data_cfg.get("dataset_root", splits_dir))
    if not dataset_root_cfg.is_absolute():
        dataset_root_cfg = Path(__file__).resolve().parent.parent / dataset_root_cfg
    dataset_root = dataset_root_cfg.resolve()
    
    # Convert labels if needed before training
    print("\n🔄 Label format kontrolü ve dönüştürme:")
    convert_labels_for_detection(splits_dir, labels_root, dataset_root)

    print("\n📦 Dataset oluşturuluyor...")
    
    # Get augmentation config
    aug_cfg = cfg.get("augmentation", {})
    augment_train = aug_cfg.get("augment", False)
    augment_val = aug_cfg.get("augment_val", False)  # Usually False for validation
    # Only keep keys supported by AugmentationFactory.get_detection_transforms
    allowed_aug_keys = {
        "horizontal_flip",
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "blur",
        "noise",
        "rotation",
        "scale",
        "translation",
        "crop_scale",
        "mixup",
        "cutmix",
        "mosaic",
    }
    aug_cfg_filtered = {
        k: v for k, v in aug_cfg.items()
        if k in allowed_aug_keys
    }
    
    print(f"   🎨 Augmentation (train): {'✅ Aktif' if augment_train else '❌ Pasif'}")
    print(f"   🎨 Augmentation (val): {'✅ Aktif' if augment_val else '❌ Pasif'}")
    
    # Create transforms
    img_size = common.get("img_size", 640)
    train_transform = AugmentationFactory.get_detection_transforms(
        augment=augment_train,
        img_size=img_size,
        **aug_cfg_filtered
    )
    val_transform = AugmentationFactory.get_detection_transforms(
        augment=augment_val,
        img_size=img_size,
        **aug_cfg_filtered
    )
    
    train_ds = DetectionDataset(train_list, labels_root, dataset_root=dataset_root, transforms=train_transform)
    val_ds = DetectionDataset(val_list, labels_root, dataset_root=dataset_root, transforms=val_transform)
    print(f"   ✅ Train dataset: {len(train_ds)} örnek")
    print(f"   ✅ Val dataset: {len(val_ds)} örnek")

    print("\n🔄 DataLoader oluşturuluyor...")
    train_loader = DataLoader(
        train_ds,
        batch_size=common["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows'ta multiprocessing sorunları olabilir
        collate_fn=collate_detection,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=common["batch_size"],
        shuffle=False,
        num_workers=0,  # Windows'ta multiprocessing sorunları olabilir
        collate_fn=collate_detection,
    )
    print(f"   ✅ Train batches: {len(train_loader)}")
    print(f"   ✅ Val batches: {len(val_loader)}")

    # num_classes here excludes background; FasterRCNN adds background internally
    print("\n🤖 Model oluşturuluyor...")
    classes_file = labels_root / "classes.txt"
    class_names = []
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if not parts:
                    continue
                if len(parts) == 2:
                    class_names.append(parts[1])
                else:
                    class_names.append(parts[0])
    if not class_names:
        class_names = ["object"]
    num_classes = len(class_names)
    model = create_detection_model(model_name, num_classes + 1)
    print(f"   ✅ Model: {model_name} (num_classes={num_classes + 1})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   🔧 Device: {device}")
    if device.type == "cuda":
        print(f"   🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   🔧 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    model.to(device)
    print(f"   ✅ Model device'a taşındı")

    # Pin memory is only meaningful on CUDA; used for non_blocking transfers below
    pin_memory = device.type == "cuda"

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params, lr=common["lr"], weight_decay=common["weight_decay"]
    )
    
    # Mixed precision training (GPU varsa)
    use_amp = AMP_AVAILABLE and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print(f"   ✅ Mixed precision training aktif (AMP)")

    base_output_dir = Path(common["output_dir"])
    preview_samples = int(common.get("preview_samples", 0))
    run_name = cfg.get("run_name", "default_run")
    safe_run_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_name))
    
    # Create run-specific output directory
    output_dir = base_output_dir / safe_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Model kayıt dizini: {output_dir}")
    
    # Early stopping and best model tracking
    early_stopping_cfg = common.get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    patience = early_stopping_cfg.get("patience", 5)
    min_delta = early_stopping_cfg.get("min_delta", 0.001)
    monitor_metric = early_stopping_cfg.get("monitor", "f1")  # For detection, monitor F1
    
    overfitting_cfg = common.get("overfitting_detection", {})
    overfitting_enabled = overfitting_cfg.get("enabled", False)
    loss_gap_threshold = overfitting_cfg.get("loss_gap_threshold", 0.5)
    metric_degradation_threshold = overfitting_cfg.get("metric_degradation_threshold", 0.05)
    
    # Best model tracking
    best_f1 = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = output_dir / "detection_best.pt"
    patience_counter = 0
    previous_metric = None
    
    if early_stopping_enabled:
        print(f"\n⏸️ Early stopping aktif: patience={patience}, monitor={monitor_metric}, min_delta={min_delta}")
    if overfitting_enabled:
        print(f"🔍 Overfitting detection aktif: loss_gap_threshold={loss_gap_threshold}")

    iou_threshold = 0.5

    print("\n" + "=" * 80)
    print("🎯 TRAINING BAŞLADI")
    print("=" * 80)
    
    def _sum_losses(loss_out):
        # Torchvision detection usually returns a dict; some variants may return list/tuple of dicts/tensors
        def _to_scalar(t):
            t = torch.as_tensor(t, device=device)
            if t.ndim == 0:
                return t.float()
            return t.float().mean()

        if isinstance(loss_out, dict):
            return sum(_to_scalar(v) for v in loss_out.values())
        if isinstance(loss_out, (list, tuple)):
            total = torch.tensor(0.0, device=device)
            for item in loss_out:
                if isinstance(item, dict):
                    total = total + sum(_to_scalar(v) for v in item.values())
                else:
                    total = total + _to_scalar(item)
            return total
        return _to_scalar(loss_out)

    for epoch in range(common["epochs"]):
        epoch_start = time.time()
        print(f"\n📊 Epoch {epoch+1}/{common['epochs']}")
        print("-" * 80)
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device, non_blocking=pin_memory) for img in images]
            targets = [{k: v.to(device, non_blocking=pin_memory) for k, v in t.items()} for t in targets]
            
            # Mixed precision training
            if use_amp:
                with autocast():
                    loss_dict = model(images, targets)
                    losses = _sum_losses(loss_dict)
                
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                losses = _sum_losses(loss_dict)
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            
            running_loss += losses.item()
            
            # Progress mesajları - her 20 batch'te bir
            if (batch_idx + 1) % 20 == 0:
                print(f"   ⏳ Batch {batch_idx+1}/{len(train_loader)} - Loss: {losses.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start
        print(f"[Detection] Epoch {epoch+1}/{common['epochs']} - Loss: {avg_loss:.4f} | Süre: {epoch_time:.1f}s")
        mlflow.log_metric("train_loss_detection", avg_loss, step=epoch)
        mlflow.log_metric("epoch_time_detection", epoch_time, step=epoch)

        # Validation loop (loss + simple IoU-based metrics) - optimized
        model.eval()
        val_loss = 0.0
        tp = 0
        fp = 0
        fn = 0
        with torch.no_grad():
            for val_batch_idx, (images, targets) in enumerate(val_loader):
                # Progress sadece her 10 batch'te bir
                if (val_batch_idx + 1) % 10 == 0:
                    print(f"   ⏳ Validation batch {val_batch_idx+1}/{len(val_loader)}")
                images = [img.to(device, non_blocking=pin_memory) for img in images]
                targets = [{k: v.to(device, non_blocking=pin_memory) for k, v in t.items()} for t in targets]
                
                # Mixed precision inference
                if use_amp:
                    with autocast():
                        loss_dict = model(images, targets)
                        losses = _sum_losses(loss_dict)
                        preds = model(images)
                else:
                    loss_dict = model(images, targets)
                    losses = _sum_losses(loss_dict)
                    preds = model(images)
                
                val_loss += losses.item()
                for pred, tgt in zip(preds, targets):
                    if tgt["boxes"].numel() == 0 and pred["boxes"].numel() == 0:
                        continue
                    if pred["boxes"].numel() == 0:
                        fn += tgt["boxes"].shape[0]
                        continue
                    if tgt["boxes"].numel() == 0:
                        fp += pred["boxes"].shape[0]
                        continue

                    # Optimized IoU calculation
                    ious = box_iou(pred["boxes"], tgt["boxes"])
                    # For simplicity: greedy matching on IoU only
                    max_iou, tgt_idx = ious.max(dim=1)
                    pred_labels = pred["labels"]
                    tgt_labels = tgt["labels"][tgt_idx]
                    matches = (max_iou >= iou_threshold) & (pred_labels == tgt_labels)
                    tp += matches.sum().item()
                    fp += (~matches).sum().item()
                    # FN: gt that are not matched - optimized
                    matched_gt = torch.zeros(tgt["boxes"].shape[0], dtype=torch.bool, device=device)
                    matched_gt[tgt_idx[matches]] = True
                    fn += (~matched_gt).sum().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(f"\n   ✅ Validation tamamlandı - Loss: {avg_val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # Overfitting detection
        loss_gap = avg_loss - avg_val_loss
        if overfitting_enabled:
            if loss_gap > loss_gap_threshold:
                print(f"   ⚠️ OVERFITTING UYARISI: Train loss - Val loss = {loss_gap:.4f} > {loss_gap_threshold}")
                mlflow.log_metric("overfitting_warning_detection", 1, step=epoch)
            else:
                mlflow.log_metric("overfitting_warning_detection", 0, step=epoch)
            
            mlflow.log_metric("loss_gap_detection", loss_gap, step=epoch)
        
        # Best model tracking (based on F1 score)
        improved = False
        if monitor_metric == "f1":
            if f1 > best_f1 + min_delta:
                best_f1 = f1
                best_val_loss = avg_val_loss
                best_epoch = epoch
                improved = True
                patience_counter = 0
            else:
                patience_counter += 1
        elif monitor_metric == "val_loss":
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_f1 = f1
                best_epoch = epoch
                improved = True
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Save best model
        if improved:
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(str(best_model_path))
            print(f"   💾 En iyi model güncellendi (F1: {f1:.4f}, Val Loss: {avg_val_loss:.4f}): {best_model_path}")
            mlflow.log_metric("best_f1_detection", best_f1, step=epoch)
            mlflow.log_metric("best_epoch_detection", best_epoch, step=epoch)
        else:
            print(f"   📊 En iyi model: Epoch {best_epoch+1} (F1: {best_f1:.4f}, Val Loss: {best_val_loss:.4f})")
            if early_stopping_enabled:
                print(f"   ⏳ Patience: {patience_counter}/{patience}")
        
        # Metric degradation detection
        if overfitting_enabled and previous_metric is not None:
            metric_drop = previous_metric - f1 if monitor_metric == "f1" else avg_val_loss - previous_metric
            if metric_drop > metric_degradation_threshold:
                print(f"   ⚠️ METRIC DEGRADATION: {monitor_metric} dropped by {metric_drop:.4f}")
                mlflow.log_metric("metric_degradation_detection", 1, step=epoch)
            else:
                mlflow.log_metric("metric_degradation_detection", 0, step=epoch)
        
        previous_metric = f1 if monitor_metric == "f1" else avg_val_loss
        
        # Batch MLflow logging (daha hızlı)
        mlflow.log_metrics({
            "val_loss_detection": avg_val_loss,
            "precision_50_detection": precision,
            "recall_50_detection": recall,
            "f1_50_detection": f1
        }, step=epoch)
        
        # Early stopping check
        if early_stopping_enabled and patience_counter >= patience:
            print(f"\n{'='*80}")
            print(f"⏸️ EARLY STOPPING: {patience} epoch boyunca iyileşme olmadı")
            print(f"   En iyi model: Epoch {best_epoch+1} (F1: {best_f1:.4f}, Val Loss: {best_val_loss:.4f})")
            print(f"{'='*80}")
            mlflow.log_param("early_stopped", True)
            mlflow.log_param("best_epoch", best_epoch + 1)
            break

    # Load best model for previews and final save
    if best_model_path.exists() and best_epoch < common["epochs"] - 1:
        print(f"\n📥 En iyi model yükleniyor (Epoch {best_epoch+1})...")
        model.load_state_dict(torch.load(best_model_path))
        print(f"   ✅ En iyi model yüklendi")
    
    if preview_samples > 0:
        print(f"\n🖼️ Örnek detection çıktıları hazırlanıyor (val'den {preview_samples} görsel)...")
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        saved = 0
        score_thresh = 0.25
        with torch.no_grad():
            for images, targets in val_loader:
                images_gpu = [img.to(device, non_blocking=pin_memory) for img in images]
                preds = model(images_gpu)
                for img_t, pred, tgt in zip(images_gpu, preds, targets):
                    if saved >= preview_samples:
                        break
                    img_np = _denorm_image(img_t)
                    vis = Image.fromarray(img_np)
                    draw = ImageDraw.Draw(vis)
                    # Ground truth (green)
                    if "boxes" in tgt and tgt["boxes"].numel() > 0:
                        for box in tgt["boxes"].cpu().numpy():
                            draw.rectangle(box.tolist(), outline=(0, 255, 0), width=2)
                    # Predictions (red)
                    boxes = pred.get("boxes", torch.empty((0, 4), device=device))
                    labels = pred.get("labels", torch.empty((0,), device=device))
                    scores = pred.get("scores", None)
                    if scores is not None:
                        keep = scores >= score_thresh
                        boxes = boxes[keep]
                        labels = labels[keep]
                        scores = scores[keep]
                    boxes = boxes.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    scores_np = scores.detach().cpu().numpy() if scores is not None else None
                    for b_idx, box in enumerate(boxes):
                        label_idx = int(labels[b_idx]) - 1  # remove background offset
                        name = class_names[label_idx] if 0 <= label_idx < len(class_names) else f"cls{label_idx}"
                        txt = name
                        if scores_np is not None:
                            txt = f"{name} {scores_np[b_idx]:.2f}"
                        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
                        draw.text((box[0] + 2, box[1] + 2), txt, fill=(255, 0, 0))
                    out_path = preview_dir / f"detection_preview_{saved+1}.png"
                    vis.save(out_path)
                    mlflow.log_artifact(str(out_path))
                    print(f"      ✅ Kaydedildi: {out_path}")
                    saved += 1
                if saved >= preview_samples:
                    break

    # Save final model (best or last)
    model_path = output_dir / "detection_last.pt"
    torch.save(model.state_dict(), model_path)
    # Log both raw artifact and MLflow model (enables model versioning / registry later)
    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(
        model, artifact_path="detection_model", registered_model_name=None
    )
    
    # Log final summary
    print(f"\n{'='*80}")
    print("📊 EĞİTİM ÖZETİ")
    print(f"{'='*80}")
    print(f"   En iyi F1 Score: {best_f1:.4f}")
    print(f"   En iyi Val Loss: {best_val_loss:.4f}")
    print(f"   En iyi Epoch: {best_epoch + 1}")
    print(f"   Best model: {best_model_path}")
    print(f"{'='*80}")
    mlflow.log_param("final_best_f1", best_f1)
    mlflow.log_param("final_best_val_loss", best_val_loss)
    mlflow.log_param("final_best_epoch", best_epoch + 1)


def train_segmentation(cfg: Dict, config_path: Path):
    print("=" * 80)
    print("🚀 SEGMENTATION TRAINING BAŞLADI")
    print("=" * 80)
    
    data_cfg = cfg["data"]
    common = cfg["common"]
    seg_cfg = cfg.get("segmentation", {})
    run_name = cfg.get("run_name", "default_run")
    safe_run_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_name))

    model_name = seg_cfg.get("model_name", "deeplabv3_resnet50")
    print(f"📋 Model: {model_name}")
    print(f"📋 Epochs: {common.get('epochs', 'N/A')}")
    print(f"📋 Batch size: {common.get('batch_size', 'N/A')}")
    print(f"📋 Learning rate: {common.get('lr', 'N/A')}")

    print("\n📂 Dataset yapılandırması:")
    splits_dir = Path(data_cfg["splits_dir"]).resolve()
    train_list = splits_dir / "train.txt"
    val_list = splits_dir / "val.txt"
    print(f"   Splits dizini: {splits_dir}")
    print(f"   Train list: {train_list} ({'✅ Var' if train_list.exists() else '❌ Yok'})")
    print(f"   Val list: {val_list} ({'✅ Var' if val_list.exists() else '❌ Yok'})")

    masks_root = Path(data_cfg["segmentation_masks_root"]).resolve()
    print(f"   Masks root: {masks_root}")
    
    # Use dataset_root from config (paths in train.txt are relative to dataset_root)
    dataset_root = Path(data_cfg.get("dataset_root", "")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path(__file__).resolve().parent.parent / dataset_root).resolve()
    
    # Heuristic: pick the folder that actually contains images_all
    candidates = [
        dataset_root,
        dataset_root / "RcCArDataset",
        Path(__file__).resolve().parent.parent / "DataSet" / "RcCArDataset",
    ]
    for cand in candidates:
        if (cand / "images_all").exists():
            dataset_root = cand.resolve()
            break
    else:
        if dataset_root.exists():
            dataset_root = dataset_root.resolve()
    
    print(f"   Dataset root: {dataset_root}")
    
    # Convert masks if needed before training
    print("\n🔄 Mask format kontrolü ve dönüştürme:")
    convert_masks_for_segmentation(splits_dir, masks_root, dataset_root)

    print("\n📦 Dataset oluşturuluyor...")
    
    # Get augmentation config
    aug_cfg = cfg.get("augmentation", {})
    augment_train = aug_cfg.get("augment", False)
    augment_val = aug_cfg.get("augment_val", False)  # Usually False for validation
    # Filter out detection-specific params (crop_scale, mixup, cutmix, mosaic) for segmentation
    seg_params = {k: v for k, v in aug_cfg.items() if k not in ("augment", "augment_val", "crop_scale", "mixup", "cutmix", "mosaic")}
    
    print(f"   🎨 Augmentation (train): {'✅ Aktif' if augment_train else '❌ Pasif'}")
    print(f"   🎨 Augmentation (val): {'✅ Aktif' if augment_val else '❌ Pasif'}")
    
    # Create transforms
    img_size = common.get("img_size", 640)
    train_transform = AugmentationFactory.get_segmentation_transforms(
        augment=augment_train,
        img_size=img_size,
        **seg_params
    )
    val_transform = AugmentationFactory.get_segmentation_transforms(
        augment=augment_val,
        img_size=img_size,
        **seg_params
    )
    
    train_ds = SegmentationDataset(train_list, masks_root, dataset_root=dataset_root, transforms=train_transform, split_name="train")
    val_ds = SegmentationDataset(val_list, masks_root, dataset_root=dataset_root, transforms=val_transform, split_name="val")

    # DataLoader optimizasyonları
    use_cuda = torch.cuda.is_available()
    num_workers = 0  # Windows'ta pickle hatası önleme; GPU hızlı, gerek yok
    pin_memory = use_cuda  # GPU varsa pin_memory kullan
    
    train_loader = DataLoader(
        train_ds,
        batch_size=common["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # avoid batch-size 1 which breaks BatchNorm in some models
        persistent_workers=num_workers > 0,  # Worker'ları canlı tut
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch optimization
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=common["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # classes.txt: first line is background, others are classes
    print("\n📋 Sınıf sayısı belirleniyor...")
    classes_file = masks_root / "classes.txt"

    def _load_classes(path: Path) -> int | None:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return len([line for line in f if line.strip()])
        return None

    num_classes = _load_classes(classes_file)
    if num_classes:
        print(f"   ✅ classes.txt bulundu: {num_classes} sınıf")
    else:
        # Fallback: check split label dirs (train/labels/classes.txt)
        alt_classes = masks_root / "train" / "labels" / "classes.txt"
        num_classes = _load_classes(alt_classes)
        if num_classes:
            print(f"   ✅ classes.txt bulundu: {alt_classes} (sınıf={num_classes})")
        else:
            # Last resort: build label map from JSONs under splits
            train_labels_dir = masks_root / "train" / "labels"
            val_labels_dir = masks_root / "val" / "labels"
            candidate_dirs = [p for p in [train_labels_dir, val_labels_dir] if p.exists()]
            label_map = collect_seg_label_map(candidate_dirs)
            num_classes = max(1, len(label_map) + 1)  # +1 for background
            print(f"   ⚠️ classes.txt yok, JSON'lardan hesaplandı: {num_classes} sınıf")
            # Save classes.txt at masks_root for future runs
            classes_file.parent.mkdir(parents=True, exist_ok=True)
            with open(classes_file, "w", encoding="utf-8") as f:
                f.write("0 background\n")
                for name, cid in sorted(label_map.items(), key=lambda x: x[1]):
                    f.write(f"{cid} {name}\n")
            print(f"   📝 classes.txt oluşturuldu: {classes_file}")

    # Load class names for metrics/visualization
    class_names: list[str] = []
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if not parts:
                    continue
                if len(parts) == 2:
                    class_names.append(parts[1])
                else:
                    class_names.append(parts[0])
    if not class_names:
        class_names = ["background"] + [f"class_{i}" for i in range(1, num_classes)]
    if len(class_names) < num_classes:
        extra = [f"class_{i}" for i in range(len(class_names), num_classes)]
        class_names.extend(extra)

    print("\n🤖 Model oluşturuluyor...")
    model = create_segmentation_model(model_name, num_classes)
    print(f"   ✅ Model: {model_name} (num_classes={num_classes})")
    
    # Check if MMSEG model (requires custom training loop)
    if isinstance(model, dict) and model.get("type") == "mmseg":
        raise NotImplementedError("MMSegmentation models require custom training loop. Coming soon.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   🔧 Device: {device}")
    if device.type == "cuda":
        print(f"   🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   🔧 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    model.to(device)
    print(f"   ✅ Model device'a taşındı")

    # Check if model returns dict (torchvision) or tensor (SMP/timm)
    is_torchvision_model = model_name in ["deeplabv3_resnet50", "fcn_resnet50"]

    # Ignore unlabeled pixels (255) in loss
    ignore_index = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.AdamW(
        model.parameters(), lr=common["lr"], weight_decay=common["weight_decay"]
    )
    
    # Mixed precision training (GPU varsa)
    use_amp = AMP_AVAILABLE and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print(f"   ✅ Mixed precision training aktif (AMP)")

    base_output_dir = Path(common["output_dir"])
    
    # Get run name for directory structure
    run_name = cfg.get("run_name", "default_run")
    safe_run_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_name))
    
    # Create run-specific output directory
    output_dir = base_output_dir / safe_run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Model kayıt dizini: {output_dir}")
    
    # Early stopping and best model tracking
    early_stopping_cfg = common.get("early_stopping", {})
    early_stopping_enabled = early_stopping_cfg.get("enabled", False)
    patience = early_stopping_cfg.get("patience", 5)
    min_delta = early_stopping_cfg.get("min_delta", 0.001)
    monitor_metric = early_stopping_cfg.get("monitor", "miou")  # For segmentation, monitor mIoU
    
    overfitting_cfg = common.get("overfitting_detection", {})
    overfitting_enabled = overfitting_cfg.get("enabled", False)
    loss_gap_threshold = overfitting_cfg.get("loss_gap_threshold", 0.5)
    metric_degradation_threshold = overfitting_cfg.get("metric_degradation_threshold", 0.05)
    
    # Best model tracking
    best_miou = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = output_dir / "segmentation_best.pt"
    patience_counter = 0
    previous_metric = None
    
    if early_stopping_enabled:
        print(f"\n⏸️ Early stopping aktif: patience={patience}, monitor={monitor_metric}, min_delta={min_delta}")
    if overfitting_enabled:
        print(f"🔍 Overfitting detection aktif: loss_gap_threshold={loss_gap_threshold}")

    print("\n" + "=" * 80, flush=True)
    print("🎯 TRAINING BAŞLADI", flush=True)
    print("=" * 80, flush=True)
    
    for epoch in range(common["epochs"]):
        epoch_start = time.time()
        print(f"\n{'='*80}", flush=True)
        print(f"📊 EPOCH {epoch+1}/{common['epochs']} - Training başladı", flush=True)
        print(f"{'='*80}", flush=True)
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            try:
                images = images.to(device, non_blocking=pin_memory)
                masks = masks.to(device, non_blocking=pin_memory)
                # Normalize mask shape to (B,H,W)
                if masks.ndim == 4:
                    if masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    elif masks.shape[-1] == 1:
                        masks = masks.squeeze(-1)

                optimizer.zero_grad()
                
                # Mixed precision training
                if use_amp:
                    with autocast():
                        model_output = model(images)
                        if isinstance(model_output, dict):
                            outputs = model_output["out"]
                        else:
                            outputs = model_output
                        loss = criterion(outputs, masks.long())
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    model_output = model(images)
                    if isinstance(model_output, dict):
                        outputs = model_output["out"]
                    else:
                        outputs = model_output
                    loss = criterion(outputs, masks.long())
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
                
                # Progress mesajları - her batch log (GPU hızlı)
                if (batch_idx + 1) % 1 == 0:
                    print(f"   ⏳ Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)
                    
            except Exception as e:
                print(f"   ❌ Batch {batch_idx+1} sırasında hata: {e}")
                import traceback
                traceback.print_exc()
                raise

        avg_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start
        print(f"\n   ✅ Training tamamlandı - Loss: {avg_loss:.4f} | Süre: {epoch_time:.1f}s", flush=True)
        # MLflow logging - sadece önemli metrikler
        mlflow.log_metric("train_loss_segmentation", avg_loss, step=epoch)
        mlflow.log_metric("epoch_time_segmentation", epoch_time, step=epoch)

        # Validation loop (loss + mIoU & pixel accuracy) - optimized
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_pixels = 0
        per_image_miou_sum = 0.0
        per_image_count = 0
        # confusion matrix for mIoU - accumulate on CPU to reduce GPU memory transfers
        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        with torch.no_grad():
            for val_batch_idx, (images, masks) in enumerate(val_loader):
                # Progress sadece her 10 batch'te bir
                if (val_batch_idx + 1) % 10 == 0:
                    print(f"   ⏳ Validation batch {val_batch_idx+1}/{len(val_loader)}")
                images = images.to(device)
                masks = masks.to(device)
                # Normalize mask shape to (B,H,W)
                if masks.ndim == 4:
                    if masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    elif masks.shape[-1] == 1:
                        masks = masks.squeeze(-1)
                model_output = model(images)
                
                # Handle different output formats
                if isinstance(model_output, dict):
                    outputs = model_output["out"]
                else:
                    outputs = model_output
                loss = criterion(outputs, masks.long())
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)

                # Valid pixels (ignore unlabeled)
                valid_mask = (masks >= 0) & (masks < num_classes) & (masks != ignore_index)
                if valid_mask.any():
                    total_correct += (preds[valid_mask] == masks[valid_mask]).sum().item()
                    total_pixels += valid_mask.sum().item()

                    # Confusion update on valid pixels only
                    masks_valid = masks[valid_mask].cpu()
                    preds_valid = preds[valid_mask].cpu()
                    inds = num_classes * masks_valid.view(-1) + preds_valid.view(-1)
                    conf_mat += torch.bincount(
                        inds, minlength=num_classes ** 2
                    ).reshape(num_classes, num_classes)

                # Per-image mIoU (unweighted per image)
                for b in range(images.shape[0]):
                    mb = masks[b]
                    pb = preds[b]
                    vm = (mb >= 0) & (mb < num_classes) & (mb != ignore_index)
                    if not vm.any():
                        continue
                    gt = mb[vm].cpu()
                    pr = pb[vm].cpu()
                    idx = num_classes * gt.view(-1) + pr.view(-1)
                    cm = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).float()
                    tp_b = torch.diag(cm)
                    fp_b = cm.sum(dim=0) - tp_b
                    fn_b = cm.sum(dim=1) - tp_b
                    denom_b = tp_b + fp_b + fn_b
                    iou_b = torch.where(denom_b > 0, tp_b / denom_b, torch.zeros_like(denom_b))
                    per_image_miou_sum += iou_b.mean().item()
                    per_image_count += 1
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"\n   ✅ Validation tamamlandı - Loss: {avg_val_loss:.4f}", end="")
        
        pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
        
        # mIoU - optimized calculation
        conf = conf_mat.float()
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        denom = tp + fp + fn
        iou_per_class = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
        miou = iou_per_class.mean().item()
        
        per_image_miou = (per_image_miou_sum / per_image_count) if per_image_count > 0 else 0.0
        print(f" | Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f} | mIoU/img: {per_image_miou:.4f}")
        
        # Debug: Print confusion matrix info
        print(f"   🔍 Confusion matrix diag: {tp.tolist()}")
        print(f"   🔍 Valid pixels per class (TP+FP+FN): {denom.tolist()}")
        
        # Overfitting detection
        loss_gap = avg_loss - avg_val_loss
        if overfitting_enabled:
            if loss_gap > loss_gap_threshold:
                print(f"   ⚠️ OVERFITTING UYARISI: Train loss - Val loss = {loss_gap:.4f} > {loss_gap_threshold}")
                mlflow.log_metric("overfitting_warning_segmentation", 1, step=epoch)
            else:
                mlflow.log_metric("overfitting_warning_segmentation", 0, step=epoch)
            
            mlflow.log_metric("loss_gap_segmentation", loss_gap, step=epoch)
        
        # Per-class IoU logging
        metrics_dict = {
            "val_loss_segmentation": avg_val_loss,
            "pixel_acc_segmentation": pixel_acc,
            "miou_segmentation": miou,
            "miou_per_image_segmentation": per_image_miou
        }
        
        # Log per-class IoU
        print("   📊 Per-class IoU:", flush=True)
        for i, iou_val in enumerate(iou_per_class):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            safe_name = class_name.replace(" ", "_")
            metrics_dict[f"iou_class_{safe_name}_segmentation"] = iou_val.item()
            print(f"      {class_name}: {iou_val.item():.4f}", flush=True)
        
        # Best model tracking (based on mIoU or val_loss)
        improved = False
        if monitor_metric == "miou":
            if miou > best_miou + min_delta:
                best_miou = miou
                best_val_loss = avg_val_loss
                best_epoch = epoch
                improved = True
                patience_counter = 0
            else:
                patience_counter += 1
        elif monitor_metric == "val_loss":
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_miou = miou
                best_epoch = epoch
                improved = True
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Save best model
        if improved:
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(str(best_model_path))
            print(f"   💾 En iyi model güncellendi (mIoU: {miou:.4f}, Val Loss: {avg_val_loss:.4f}): {best_model_path}")
            mlflow.log_metric("best_miou_segmentation", best_miou, step=epoch)
            mlflow.log_metric("best_epoch_segmentation", best_epoch, step=epoch)
        else:
            print(f"   📊 En iyi model: Epoch {best_epoch+1} (mIoU: {best_miou:.4f}, Val Loss: {best_val_loss:.4f})")
            if early_stopping_enabled:
                print(f"   ⏳ Patience: {patience_counter}/{patience}")
        
        # Metric degradation detection
        if overfitting_enabled and previous_metric is not None:
            metric_drop = previous_metric - miou if monitor_metric == "miou" else avg_val_loss - previous_metric
            if metric_drop > metric_degradation_threshold:
                print(f"   ⚠️ METRIC DEGRADATION: {monitor_metric} dropped by {metric_drop:.4f}")
                mlflow.log_metric("metric_degradation_segmentation", 1, step=epoch)
            else:
                mlflow.log_metric("metric_degradation_segmentation", 0, step=epoch)
        
        previous_metric = miou if monitor_metric == "miou" else avg_val_loss
        
        # Batch MLflow logging (daha hızlı)
        mlflow.log_metrics(metrics_dict, step=epoch)
        
        # Early stopping check
        if early_stopping_enabled and patience_counter >= patience:
            print(f"\n{'='*80}")
            print(f"⏸️ EARLY STOPPING: {patience} epoch boyunca iyileşme olmadı")
            print(f"   En iyi model: Epoch {best_epoch+1} (mIoU: {best_miou:.4f}, Val Loss: {best_val_loss:.4f})")
            print(f"{'='*80}")
            mlflow.log_param("early_stopped", True)
            mlflow.log_param("best_epoch", best_epoch + 1)
            break

    # Load best model for previews and final save
    if best_model_path.exists() and best_epoch < common["epochs"] - 1:
        print(f"\n📥 En iyi model yükleniyor (Epoch {best_epoch+1})...")
        model.load_state_dict(torch.load(best_model_path))
        print(f"   ✅ En iyi model yüklendi")
    
    preview_samples = int(common.get("preview_samples", 0))
    if preview_samples > 0:
        print(f"\n🖼️ Örnek segmentation çıktıları hazırlanıyor (val'den {preview_samples} görsel)...")
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        palette = _build_palette(num_classes)
        saved = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                model_output = model(images)
                if isinstance(model_output, dict):
                    outputs = model_output.get("out", model_output.get("logits", None))
                else:
                    outputs = model_output
                if outputs is None:
                    continue
                preds = outputs.argmax(dim=1).cpu()
                imgs_np = images.cpu()
                for b in range(imgs_np.shape[0]):
                    if saved >= preview_samples:
                        break
                    img_rgb = _denorm_image(imgs_np[b])
                    gt = masks[b].detach().cpu().numpy()
                    pred = preds[b].numpy()
                    # Ensure 2D masks (H, W)
                    if gt.ndim == 3:
                        gt = gt.squeeze()
                    if pred.ndim == 3:
                        pred = pred.squeeze()
                    gt_color = palette[gt]
                    pred_color = palette[pred]
                    # Apply semi-transparent overlays on the original image
                    overlay_gt = (0.6 * img_rgb + 0.4 * gt_color).astype(np.uint8)
                    overlay_pred = (0.6 * img_rgb + 0.4 * pred_color).astype(np.uint8)
                    grid = np.concatenate([img_rgb, overlay_gt, overlay_pred], axis=1)
                    out_path = preview_dir / f"segmentation_preview_{saved+1}.png"
                    Image.fromarray(grid).save(out_path)
                    mlflow.log_artifact(str(out_path))
                    print(f"      ✅ Kaydedildi: {out_path}")
                    saved += 1
                if saved >= preview_samples:
                    break

    print("\n" + "=" * 80)
    print("💾 Model kaydediliyor...")
    print("=" * 80)
    model_path = output_dir / "segmentation_last.pt"
    torch.save(model.state_dict(), model_path)
    print(f"   ✅ Model kaydedildi: {model_path}")
    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(
        model, artifact_path="segmentation_model", registered_model_name=None
    )
    print("   ✅ MLflow'a model kaydedildi")
    
    # Log final summary
    print(f"\n{'='*80}")
    print("📊 EĞİTİM ÖZETİ")
    print(f"{'='*80}")
    print(f"   En iyi mIoU: {best_miou:.4f}")
    print(f"   En iyi Val Loss: {best_val_loss:.4f}")
    print(f"   En iyi Epoch: {best_epoch + 1}")
    print(f"   Best model: {best_model_path}")
    print(f"{'='*80}")
    mlflow.log_param("final_best_miou", best_miou)
    mlflow.log_param("final_best_val_loss", best_val_loss)
    mlflow.log_param("final_best_epoch", best_epoch + 1)
    
    print("\n" + "=" * 80)
    print("✅ SEGMENTATION TRAINING TAMAMLANDI")
    print("=" * 80)
    
    # Run evaluation and save results to test_outputs
    print("\n" + "=" * 80)
    print("🔍 EVALUATION BAŞLANIYOR")
    print("=" * 80)
    
    # Reload best model for evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    eval_miou = evaluate_segmentation(
        model, 
        val_loader, 
        num_classes, 
        class_names, 
        device, 
        output_dir="test_outputs",
        run_name=safe_run_name,
        num_samples=None  # Evaluate all samples
    )
    
    print(f"\n   📊 Evaluation mIoU: {eval_miou:.4f}")
    mlflow.log_metric("evaluation_miou", eval_miou)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION TAMAMLANDI")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Multi-task training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    print(f"📄 Config dosyası: {config_path} ({'✅ Var' if config_path.exists() else '❌ Yok'})")
    cfg = load_config(config_path)
    print(f"✅ Config yüklendi")

    task_type = cfg.get("task_type", "detection")
    print(f"📋 Task type: {task_type}")
    
    print("\n🔧 MLflow yapılandırması:")
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("autonomous_vehicle")
    print(f"   ✅ Tracking URI: mlruns")
    print(f"   ✅ Experiment: autonomous_vehicle")

    # Get run name from config if provided
    run_name = cfg.get("run_name", None)
    if run_name:
        print(f"📝 Run name: {run_name}")
    else:
        print(f"📝 Run name: (belirtilmedi)")
    
    print("\n🚀 MLflow run başlatılıyor...")
    with mlflow.start_run(run_name=run_name):
        # Also set as tag for better compatibility
        if run_name:
            mlflow.set_tag("mlflow.runName", run_name)
            mlflow.set_tag("experiment_name", run_name)
        
        print("   ✅ MLflow run başlatıldı")
        print("\n📊 Parametreler MLflow'a kaydediliyor...")
        # log basic params
        mlflow.log_param("task_type", task_type)
        for k, v in cfg.get("common", {}).items():
            # Skip nested dictionaries (they will be logged separately)
            if isinstance(v, dict):
                continue
            mlflow.log_param(f"common_{k}", v)
        
        # Log augmentation parameters
        if "augmentation" in cfg:
            for k, v in cfg["augmentation"].items():
                mlflow.log_param(f"augmentation_{k}", v)
        
        # Log early stopping parameters
        if "early_stopping" in cfg.get("common", {}):
            for k, v in cfg["common"]["early_stopping"].items():
                mlflow.log_param(f"early_stopping_{k}", v)
        
        # Log overfitting detection parameters
        if "overfitting_detection" in cfg.get("common", {}):
            for k, v in cfg["common"]["overfitting_detection"].items():
                mlflow.log_param(f"overfitting_{k}", v)

        if task_type == "detection":
            for k, v in cfg.get("detection", {}).items():
                mlflow.log_param(f"detection_{k}", v)
            print("   ✅ Parametreler kaydedildi")
            print("\n" + "=" * 80)
            train_detection(cfg, config_path)
        elif task_type == "segmentation":
            for k, v in cfg.get("segmentation", {}).items():
                mlflow.log_param(f"segmentation_{k}", v)
            print("   ✅ Parametreler kaydedildi")
            print("\n" + "=" * 80)
            train_segmentation(cfg, config_path)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")


if __name__ == "__main__":
    main()


