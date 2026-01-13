"""
Segmentation evaluation module integrated into training pipeline.
Generates metrics summary and visual error analysis after training.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from torch.utils.data import DataLoader

from data.dataset import SegmentationDataset


def build_palette(num_classes):
    """Build color palette for visualization."""
    base = np.array([
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
    ], dtype=np.uint8)
    if num_classes <= len(base):
        return base[:num_classes]
    reps = int(np.ceil(num_classes / len(base)))
    return np.tile(base, (reps, 1))[:num_classes]


def compute_iou_with_details(pred_mask, gt_mask, num_classes):
    """Compute IoU and return per-class details."""
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    k = (gt_mask >= 0) & (gt_mask < num_classes) & (gt_mask != 255)
    if k.any():
        gt_filtered = gt_mask[k]
        pred_filtered = pred_mask[k]
        inds = num_classes * gt_filtered.view(-1) + pred_filtered.view(-1)
        conf_mat = torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    
    conf = conf_mat.float()
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou_per_class = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    
    return iou_per_class, conf_mat


def visualize_errors(image, gt_mask, pred_mask, class_names, palette):
    """Create error visualization: green=correct, red=FP, blue=FN."""
    num_classes = len(class_names)
    iou_per_class, _ = compute_iou_with_details(
        torch.from_numpy(pred_mask),
        torch.from_numpy(gt_mask),
        num_classes
    )
    
    # Clip mask values to valid range
    gt_mask = np.clip(gt_mask, 0, num_classes - 1)
    pred_mask = np.clip(pred_mask, 0, num_classes - 1)
    
    # Create error map: red=FP, blue=FN, green=correct
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    
    for c in range(num_classes):
        correct = (gt_mask == c) & (pred_mask == c)
        false_pos = (gt_mask != c) & (pred_mask == c)
        false_neg = (gt_mask == c) & (pred_mask != c)
        
        error_map[correct] = [0, 255, 0]      # Green = correct
        error_map[false_pos] = [0, 0, 255]    # Red = false positives
        error_map[false_neg] = [255, 0, 0]    # Blue = false negatives
    
    # Create comparison grid
    gt_color = palette[gt_mask]
    pred_color = palette[pred_mask]
    
    # Normalize image
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    
    # Resize all to match
    if error_map.shape[:2] != image.shape[:2]:
        error_map = cv2.resize(error_map, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_color.shape[:2] != image.shape[:2]:
        gt_color = cv2.resize(gt_color, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_color.shape[:2] != image.shape[:2]:
        pred_color = cv2.resize(pred_color, (w, h), interpolation=cv2.INTER_NEAREST)
    
    grid = np.concatenate([
        image_uint8,
        gt_color,
        pred_color,
        error_map
    ], axis=1)
    
    return grid, iou_per_class


def evaluate_segmentation(model, val_loader, num_classes, class_names, device, output_dir, run_name, num_samples=None):
    """
    Evaluate segmentation model and save results.
    
    Args:
        model: Segmentation model
        val_loader: Validation DataLoader
        num_classes: Number of classes
        class_names: List of class names
        device: torch device
        output_dir: Base output directory (test_outputs)
        run_name: Name of the run
        num_samples: Max samples to visualize (None = all)
    """
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = torch.device(device)
    palette = build_palette(num_classes)
    
    conf_mat_total = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    class_iou_totals = {i: [] for i in range(num_classes)}
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
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
            
            preds = outputs.argmax(dim=1)
            
            for b in range(images.shape[0]):
                image = images[b].cpu().numpy()
                pred_mask = preds[b].cpu().numpy()
                gt_mask = masks[b].cpu().numpy()
                
                # Denormalize image
                image = image.transpose(1, 2, 0)
                image = ((image * 0.5) + 0.5).clip(0, 1)
                
                # Compute IoU
                iou_per_class, local_conf = compute_iou_with_details(
                    torch.from_numpy(pred_mask),
                    torch.from_numpy(gt_mask),
                    num_classes
                )
                
                # Accumulate
                for i in range(num_classes):
                    class_iou_totals[i].append(iou_per_class[i].item())
                conf_mat_total += local_conf
                
                # Visualize
                grid, _ = visualize_errors(image, gt_mask, pred_mask, class_names, palette)
                
                out_path = output_dir / f"eval_sample_{sample_count+1:03d}.png"
                Image.fromarray(grid).save(out_path)
                
                sample_count += 1
                if num_samples and sample_count >= num_samples:
                    break
            
            if num_samples and sample_count >= num_samples:
                break
    
    # Save summary
    summary_path = output_dir / "metrics_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SEGMENTATION EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-sample average IoU (unweighted):\n")
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            avg_iou = np.mean(class_iou_totals[i]) if class_iou_totals[i] else 0.0
            f.write(f"  {class_name}: {avg_iou:.4f} (n={len(class_iou_totals[i])})\n")
        
        overall_avg = np.mean([iou for iou_list in class_iou_totals.values() for iou in iou_list])
        f.write(f"\nOverall mIoU (per-sample average): {overall_avg:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("GLOBAL (pixel-weighted) metrics:\n")
        conf = conf_mat_total.float()
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        denom = tp + fp + fn
        iou_per_class_global = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
        miou_global = iou_per_class_global.mean().item()
        
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            f.write(f"  {class_name}: {iou_per_class_global[i].item():.4f}\n")
        
        f.write(f"\nGlobal mIoU (pixel-weighted): {miou_global:.4f}\n")
        f.write(f"\nTotal samples evaluated: {sample_count}\n")
    
    print(f"\n✅ Evaluation complete: {output_dir}")
    print(f"   Samples visualized: {sample_count}")
    print(f"   Summary: {summary_path}")
    
    return miou_global
