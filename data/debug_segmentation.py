"""
Debug script to analyze segmentation predictions and diagnose metric vs visual quality mismatch.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SegmentationDataset


def compute_iou_with_details(pred_mask, gt_mask, num_classes):
    """Compute IoU and return per-class details."""
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    
    k = (gt_mask >= 0) & (gt_mask < num_classes)
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
    """Create a detailed error visualization showing where model fails."""
    
    # Get class IoU for this sample
    num_classes = len(class_names)
    iou_per_class, _ = compute_iou_with_details(
        torch.from_numpy(pred_mask),
        torch.from_numpy(gt_mask),
        num_classes
    )
    
    # Clip mask values to valid range
    gt_mask = np.clip(gt_mask, 0, num_classes - 1)
    pred_mask = np.clip(pred_mask, 0, num_classes - 1)
    
    # Create error map: red = false positives, blue = false negatives, green = correct
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    
    for c in range(num_classes):
        correct = (gt_mask == c) & (pred_mask == c)
        false_pos = (gt_mask != c) & (pred_mask == c)  # Predicted this class but shouldn't
        false_neg = (gt_mask == c) & (pred_mask != c)  # Should be this class but predicted another
        
        error_map[correct] = [0, 255, 0]      # Green = correct
        error_map[false_pos] = [0, 0, 255]    # Red = false positives
        error_map[false_neg] = [255, 0, 0]    # Blue = false negatives
    
    # Create comparison grid
    gt_color = palette[gt_mask]
    pred_color = palette[pred_mask]
    
    # Normalize image if needed
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    # Create side-by-side with error map
    h, w = image.shape[:2]
    
    # Resize error map to match
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


def denorm_image(img_tensor):
    """Reverse normalization: ((x-0.5)/0.5) -> uint8."""
    img = img_tensor.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW format
        img = np.transpose(img, (1, 2, 0))
    img = ((img * 0.5) + 0.5).clip(0, 1)
    return (img * 255).astype(np.uint8)


def analyze_predictions(model, val_loader, num_classes, class_names, device, output_dir, num_samples=10):
    """
    Analyze predictions and create detailed visualizations.
    Shows: Original | GT Mask | Pred Mask | Error Map
    Error map: Green=correct, Red=false positives, Blue=false negatives
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    palette = build_palette(num_classes)
    
    # Expand palette to handle unexpected class indices
    max_palette_size = 256
    if len(palette) < max_palette_size:
        extended = np.random.randint(0, 255, (max_palette_size - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, extended])
    
    model.eval()
    
    sample_count = 0
    class_iou_totals = {i: [] for i in range(num_classes)}  # per-sample mean (unweighted)
    conf_mat_total = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # global aggregation
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.ndim == 4:
                if masks.shape[1] == 1:
                    masks = masks.squeeze(1)
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            
            preds = outputs.argmax(dim=1)
            
            for b in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                
                img_np = denorm_image(images[b])
                gt_np = masks[b].cpu().numpy()
                pred_np = preds[b].cpu().numpy()
                
                # Ensure 2D
                if gt_np.ndim == 3:
                    gt_np = gt_np.squeeze()
                if pred_np.ndim == 3:
                    pred_np = pred_np.squeeze()
                
                # Create visualizations and collect metrics
                error_grid, iou_per_class = visualize_errors(img_np, gt_np, pred_np, class_names, palette)
                
                # Track IoU for this sample
                for i in range(num_classes):
                    class_iou_totals[i].append(iou_per_class[i].item())

                # Accumulate global confusion matrix to match training computation
                _, conf_mat = compute_iou_with_details(
                    torch.from_numpy(pred_np), torch.from_numpy(gt_np), num_classes
                )
                conf_mat_total += conf_mat
                
                # Save
                out_path = output_dir / f"debug_sample_{sample_count+1:03d}.png"
                Image.fromarray(error_grid).save(out_path)
                
                # Also save a text report for this sample
                report_path = output_dir / f"debug_sample_{sample_count+1:03d}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(f"Sample {sample_count+1}\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Legend: Green=Correct | Red=False Positive | Blue=False Negative\n\n")
                    f.write("Per-class IoU:\n")
                    for i, iou_val in enumerate(iou_per_class):
                        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
                        f.write(f"  {class_name}: {iou_val.item():.4f}\n")
                
                print(f"✅ Saved: {out_path}")
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # Summary report
    summary_path = output_dir / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SEGMENTATION ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("Average IoU per class across analyzed samples (unweighted by pixels):\n")
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"class_{i}"
            avg_iou = np.mean(class_iou_totals[i]) if class_iou_totals[i] else 0.0
            f.write(f"  {class_name}: {avg_iou:.4f} (n={len(class_iou_totals[i])})\n")
        
        overall_avg = np.mean([iou for iou_list in class_iou_totals.values() for iou in iou_list])
        f.write(f"\nOverall mIoU (unweighted): {overall_avg:.4f}\n")

        # Global metrics to match training computation
        f.write("\nGLOBAL (pixel-weighted) metrics to match training:\n")
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
    
    print(f"\n✅ Analysis complete. Summary saved to: {summary_path}")


def main():
    """Main analysis function."""
    import argparse
    from data.train import load_config, create_segmentation_model
    
    parser = argparse.ArgumentParser(description="Debug segmentation predictions.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config file path")
    parser.add_argument("--model-path", type=str, help="Path to saved model weights")
    parser.add_argument("--output-dir", type=str, default="debug_output", help="Output directory for analysis")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to analyze")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    
    # Setup paths
    splits_dir = Path(data_cfg["splits_dir"]).resolve()
    val_list = splits_dir / "val.txt"
    masks_root = Path(data_cfg["segmentation_masks_root"]).resolve()
    dataset_root = Path(data_cfg.get("dataset_root", "")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (Path(__file__).resolve().parent.parent / dataset_root).resolve()
    
    # Load classes - check multiple locations
    classes_file = masks_root / "classes.txt"
    if not classes_file.exists():
        # Try train/labels/classes.txt
        classes_file = masks_root / "train" / "labels" / "classes.txt"
    if not classes_file.exists():
        # Try val/labels/classes.txt
        classes_file = masks_root / "val" / "labels" / "classes.txt"
    
    class_names = []
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) >= 2:
                    class_names.append(parts[1])
                elif parts:
                    class_names.append(parts[0])
    if not class_names:
        class_names = ["background", "object"]
    
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg.get("segmentation", {}).get("model_name", "deeplabv3_resnet50")
    model = create_segmentation_model(model_name, num_classes)
    model.to(device)
    
    if args.model_path and Path(args.model_path).exists():
        print(f"Loading model from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Load dataset
    val_ds = SegmentationDataset(val_list, masks_root, dataset_root=dataset_root, split_name="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Analyzing {args.num_samples} samples...")
    analyze_predictions(model, val_loader, num_classes, class_names, device, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
