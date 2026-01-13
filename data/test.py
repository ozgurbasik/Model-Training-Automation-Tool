"""
Test script for evaluating trained models on test dataset.
Runs as a subprocess with real-time console output.
"""
import argparse
import sys
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from PIL import Image, ImageDraw

# Import from train.py
from data.train import (
    create_detection_model,
    create_segmentation_model,
    _denorm_image,
    _build_palette,
)
from data.dataset import (
    DetectionDataset,
    SegmentationDataset,
    default_detection_transform,
    default_segmentation_transform,
)


def test_detection(args):
    """Run detection test."""
    print("\n" + "=" * 80)
    print("🧪 DETECTION TEST BAŞLADI")
    print("=" * 80)
    
    print(f"\n📋 Test parametreleri:")
    print(f"   Model path: {args.model_path}")
    print(f"   Model name: {args.model_name}")
    print(f"   Splits dir: {args.splits_dir}")
    print(f"   Test labels dir: {args.test_labels_dir}")
    print(f"   Image size: {args.img_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num classes: {args.num_classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    splits_dir = Path(args.splits_dir)
    test_labels_dir = Path(args.test_labels_dir)
    dataset_root = splits_dir
    
    # Check for YOLO model
    is_yolo = args.model_name.startswith("yolov8") or args.model_name.startswith("yolov5")
    
    if is_yolo:
        print("\n🤖 Loading YOLO model...")
        try:
            from ultralytics import YOLO
            model = YOLO(args.model_path)
            print(f"   ✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load YOLO model: {e}")
            sys.exit(1)
        
        # YOLO test uses different workflow
        return test_yolo_detection(model, args, splits_dir, test_labels_dir, device)
    
    # Load PyTorch model
    print("\n🤖 Loading detection model...")
    try:
        model = create_detection_model(args.model_name, args.num_classes + 1)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create test dataset
    print("\n📦 Creating test dataset...")
    test_dir = splits_dir / "test"
    test_images_dir = test_dir / "images"
    
    # Create temporary test list
    import tempfile
    test_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    test_images = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg"))
    for img_path in test_images:
        rel_path = img_path.relative_to(splits_dir)
        test_list_file.write(f"{rel_path}\n")
    test_list_file.close()
    test_list_path = Path(test_list_file.name)
    
    test_dataset = DetectionDataset(
        test_list_path,
        test_labels_dir,
        dataset_root=dataset_root,
        transforms=default_detection_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"   ✅ Test dataset created: {len(test_dataset)} images")
    print(f"   ✅ Test batches: {len(test_loader)}")
    
    # Run inference
    print("\n🔄 Running inference...")
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            start_time = time.time()
            predictions = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    iou_threshold = 0.5
    tp, fp, fn = 0, 0, 0
    total_gt = sum(t["boxes"].shape[0] for t in all_targets)
    total_pred = sum(p.get("boxes", torch.empty((0, 4), device=device)).shape[0] for p in all_predictions)
    
    print(f"   Total GT boxes: {total_gt}")
    print(f"   Total predicted boxes: {total_pred}")
    
    for pred, tgt in zip(all_predictions, all_targets):
        if tgt["boxes"].numel() == 0 and pred["boxes"].numel() == 0:
            continue
        if pred["boxes"].numel() == 0:
            fn += tgt["boxes"].shape[0]
            continue
        if tgt["boxes"].numel() == 0:
            fp += pred["boxes"].shape[0]
            continue
        
        ious = box_iou(pred["boxes"], tgt["boxes"])
        max_iou, tgt_idx = ious.max(dim=1)
        pred_labels = pred["labels"]
        tgt_labels = tgt["labels"][tgt_idx]
        matches = (max_iou >= iou_threshold) & (pred_labels == tgt_labels)
        tp += matches.sum().item()
        fp += (~matches).sum().item()
        
        matched_gt = torch.zeros(tgt["boxes"].shape[0], dtype=torch.bool, device=device)
        matched_gt[tgt_idx[matches]] = True
        fn += (~matched_gt).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    
    # Generate preview images if output dir provided
    if args.output_dir:
        print("\n🖼️  Generating preview images...")
        output_dir = Path(args.output_dir)
        preview_dir = output_dir / "test_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        # Load class names
        class_names = []
        classes_file = splits_dir / "train" / "labels" / "classes.txt"
        if classes_file.exists():
            with open(classes_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        class_names.append(parts[1])
                    elif parts:
                        class_names.append(parts[0])
        if not class_names:
            class_names = ["object"]
        
        saved = 0
        score_thresh = 0.25
        max_preview = min(10, len(test_dataset))
        
        test_loader_vis = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for vis_idx, (images, targets) in enumerate(test_loader_vis):
                if vis_idx >= max_preview:
                    break
                
                images_gpu = [img.to(device) for img in images]
                preds = model(images_gpu)
                
                for img_t, pred, tgt in zip(images_gpu, preds, targets):
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
                        label_idx = int(labels[b_idx]) - 1
                        name = class_names[label_idx] if 0 <= label_idx < len(class_names) else f"cls{label_idx}"
                        txt = name
                        if scores_np is not None:
                            txt = f"{name} {scores_np[b_idx]:.2f}"
                        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
                        draw.text((box[0] + 2, box[1] + 2), txt, fill=(255, 0, 0))
                    
                    if vis_idx < len(test_dataset.image_paths):
                        img_name = test_dataset.image_paths[vis_idx].stem
                        out_path = preview_dir / f"{img_name}_test.png"
                        vis.save(out_path)
                        saved += 1
        
        print(f"   ✅ Saved {saved} preview images to: {preview_dir}")
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 TEST SONUÇLARI")
    print("=" * 80)
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   True Positives: {tp}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   Average Inference Time: {avg_inference_time*1000:.2f} ms")
    print(f"   Total Images: {len(test_dataset)}")
    print("=" * 80)
    
    # Cleanup
    import os
    os.unlink(test_list_path)
    
    return 0


def test_yolo_detection(model, args, splits_dir, test_labels_dir, device):
    """Run YOLO detection test."""
    print("\n📋 Running YOLO validation...")
    
    test_dir = splits_dir / "test"
    test_images_dir = test_dir / "images"
    
    # Load class names
    class_names = []
    classes_file = splits_dir / "train" / "labels" / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    class_names.append(parts[1])
                elif parts:
                    class_names.append(parts[0])
    if not class_names:
        class_names = ["object"]
    
    # Create data.yaml for YOLO
    import yaml
    import tempfile
    
    data_yaml = {
        "path": str(splits_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    
    temp_data_yaml = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(data_yaml, temp_data_yaml)
    temp_data_yaml.close()
    
    print(f"   Data config: {temp_data_yaml.name}")
    print(f"   Classes: {class_names}")
    
    try:
        results = model.val(data=temp_data_yaml.name, device=device, split="test")
        
        print("\n" + "=" * 80)
        print("📊 YOLO TEST SONUÇLARI")
        print("=" * 80)
        
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            print(f"   Precision: {metrics.get('metrics/precision(B)', 0.0):.4f}")
            print(f"   Recall: {metrics.get('metrics/recall(B)', 0.0):.4f}")
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 0.0):.4f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0.0):.4f}")
        
        print("=" * 80)
        
    finally:
        import os
        os.unlink(temp_data_yaml.name)
    
    return 0


def test_segmentation(args):
    """Run segmentation test."""
    print("\n" + "=" * 80)
    print("🧪 SEGMENTATION TEST BAŞLADI")
    print("=" * 80)
    
    print(f"\n📋 Test parametreleri:")
    print(f"   Model path: {args.model_path}")
    print(f"   Model name: {args.model_name}")
    print(f"   Splits dir: {args.splits_dir}")
    print(f"   Image size: {args.img_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num classes: {args.num_classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model
    print("\n🤖 Loading segmentation model...")
    try:
        model = create_segmentation_model(args.model_name, args.num_classes)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create test dataset
    print("\n📦 Creating test dataset...")
    splits_dir = Path(args.splits_dir)
    test_dir = splits_dir / "test"
    test_images_dir = test_dir / "images"
    masks_root = splits_dir
    dataset_root = splits_dir
    
    # Create temporary test list
    import tempfile
    test_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    test_images = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
    for img_path in test_images:
        rel_path = img_path.relative_to(splits_dir)
        test_list_file.write(f"{rel_path}\n")
    test_list_file.close()
    test_list_path = Path(test_list_file.name)
    
    test_dataset = SegmentationDataset(
        test_list_path,
        masks_root,
        dataset_root=dataset_root,
        transforms=default_segmentation_transform,
        split_name="test"
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"   ✅ Test dataset created: {len(test_dataset)} images")
    print(f"   ✅ Test batches: {len(test_loader)}")
    
    # Run inference
    print("\n🔄 Running inference...")
    total_correct = 0
    total_pixels = 0
    conf_mat = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.ndim == 4:
                if masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                elif masks.shape[-1] == 1:
                    masks = masks.squeeze(-1)
            
            start_time = time.time()
            model_output = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            if isinstance(model_output, dict):
                outputs = model_output["out"]
            else:
                outputs = model_output
            
            preds = outputs.argmax(dim=1).cpu()
            masks_cpu = masks.cpu().long()
            
            # Clamp mask values
            masks_cpu = torch.clamp(masks_cpu, 0, args.num_classes - 1)
            
            total_correct += (preds == masks_cpu).sum().item()
            total_pixels += masks_cpu.numel()
            
            # Confusion matrix
            valid_mask = (masks_cpu >= 0) & (masks_cpu < args.num_classes)
            if valid_mask.any():
                masks_flat = masks_cpu[valid_mask].view(-1)
                preds_flat = preds[valid_mask].view(-1)
                preds_flat = torch.clamp(preds_flat, 0, args.num_classes - 1)
                inds = args.num_classes * masks_flat + preds_flat
                conf_mat += torch.bincount(inds, minlength=args.num_classes ** 2).reshape(args.num_classes, args.num_classes)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    pixel_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    
    conf = conf_mat.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou_per_class = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    miou = iou_per_class.mean().item()
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    
    # Generate preview images if output dir provided
    if args.output_dir:
        print("\n🖼️  Generating preview images...")
        output_dir = Path(args.output_dir)
        preview_dir = output_dir / "test_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        saved = 0
        palette = _build_palette(args.num_classes)
        max_preview = min(10, len(test_dataset))
        
        test_loader_vis = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for vis_idx, (images, masks) in enumerate(test_loader_vis):
                if vis_idx >= max_preview:
                    break
                
                images = images.to(device)
                masks = masks.to(device)
                
                if masks.ndim == 4:
                    if masks.shape[1] == 1:
                        masks = masks.squeeze(1)
                    elif masks.shape[-1] == 1:
                        masks = masks.squeeze(-1)
                
                model_output = model(images)
                if isinstance(model_output, dict):
                    outputs = model_output["out"]
                else:
                    outputs = model_output
                
                preds = outputs.argmax(dim=1).cpu()
                imgs_np = images.cpu()
                
                for b in range(imgs_np.shape[0]):
                    img_rgb = _denorm_image(imgs_np[b])
                    gt = masks[b].detach().cpu().numpy()
                    pred = preds[b].numpy()
                    
                    if gt.ndim == 3:
                        gt = gt.squeeze()
                    if pred.ndim == 3:
                        pred = pred.squeeze()
                    
                    gt_color = palette[gt]
                    pred_color = palette[pred]
                    
                    overlay_gt = (0.6 * img_rgb + 0.4 * gt_color).astype(np.uint8)
                    overlay_pred = (0.6 * img_rgb + 0.4 * pred_color).astype(np.uint8)
                    
                    grid = np.concatenate([img_rgb, overlay_gt, overlay_pred], axis=1)
                    
                    img_path = test_dataset.image_paths[vis_idx]
                    out_path = preview_dir / f"{img_path.stem}_test.png"
                    Image.fromarray(grid).save(out_path)
                    saved += 1
        
        print(f"   ✅ Saved {saved} preview images to: {preview_dir}")
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 TEST SONUÇLARI")
    print("=" * 80)
    print(f"   Pixel Accuracy: {pixel_acc:.4f}")
    print(f"   mIoU: {miou:.4f}")
    print(f"   Average Inference Time: {avg_inference_time*1000:.2f} ms")
    print(f"   Total Images: {len(test_dataset)}")
    print("\n   Per-Class IoU:")
    for i, iou in enumerate(iou_per_class.numpy()):
        print(f"      Class {i}: {iou:.4f}")
    print("=" * 80)
    
    # Cleanup
    import os
    os.unlink(test_list_path)
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument("--task_type", type=str, required=True, choices=["detection", "segmentation"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--model_name", type=str, required=True, help="Model architecture name")
    parser.add_argument("--splits_dir", type=str, required=True, help="Path to splits directory")
    parser.add_argument("--test_labels_dir", type=str, default="", help="Path to test labels (detection only)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for previews")
    
    args = parser.parse_args()
    
    try:
        if args.task_type == "detection":
            return test_detection(args)
        else:
            return test_segmentation(args)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
