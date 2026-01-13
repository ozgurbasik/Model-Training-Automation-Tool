"""
Segmentation model inference script.
Test görüntüleri üzerinde eğitilmiş modeli çalıştırır ve sonuçları kaydeder.
"""
import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

# Model imports
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from torchvision import transforms

# Segmentation Models PyTorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# Timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Transformers
try:
    from transformers import AutoModelForImageSegmentation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Mixed precision
try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


def create_segmentation_model(model_name: str, num_classes: int):
    """Create segmentation model based on model_name."""
    # Mask2Former
    if model_name == "mask2former":
        if TRANSFORMERS_AVAILABLE:
            try:
                model = AutoModelForImageSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic")
                model.config.num_labels = num_classes
                return model
            except:
                pass
        raise ImportError("Mask2Former requires transformers. Install: pip install transformers")
    
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
    
    # Timm models
    if TIMM_AVAILABLE:
        if model_name == "segformer":
            model = timm.create_model("segformer_b0", pretrained=True, num_classes=num_classes)
            return model
        elif model_name == "hrnet":
            model = timm.create_model("hrnet_w18", pretrained=True, num_classes=num_classes)
            return model
    
    # Torchvision models
    if model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights=None, aux_loss=True)
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # Update aux classifier if it exists
        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            aux_in_channels = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
        return model
    elif model_name == "fcn_resnet50":
        model = fcn_resnet50(weights=None, aux_loss=True)
        in_channels = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # Update aux classifier if it exists
        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            aux_in_channels = model.aux_classifier[4].in_channels
            model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
        return model
    
    raise ValueError(f"Unsupported segmentation model_name: {model_name}")


def load_model(model_path: Path, model_name: str, num_classes: int, device: torch.device):
    """Load trained segmentation model."""
    print(f"[INFO] Model yukleniyor: {model_path}")
    print(f"   Model tipi: {model_name}")
    
    # Load checkpoint first to detect num_classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Auto-detect num_classes from checkpoint
    detected_num_classes = num_classes
    if "classifier.4.weight" in state_dict:
        detected_num_classes = state_dict["classifier.4.weight"].shape[0]
        print(f"   [INFO] Sinif sayisi checkpoint'ten tespit edildi: {detected_num_classes}")
    elif "classifier.4.bias" in state_dict:
        detected_num_classes = state_dict["classifier.4.bias"].shape[0]
        print(f"   [INFO] Sinif sayisi checkpoint'ten tespit edildi: {detected_num_classes}")
    else:
        print(f"   [INFO] Sinif sayisi: {num_classes} (varsayilan)")
        detected_num_classes = num_classes
    
    # Create model architecture with detected num_classes
    model = create_segmentation_model(model_name, detected_num_classes)
    model.to(device)
    
    # Filter out keys that don't exist in the model (for compatibility)
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"   [WARN] Skipping {k} due to shape mismatch: {model_state_dict[k].shape} vs {v.shape}")
        else:
            print(f"   [WARN] Skipping {k} (not in model)")
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    print(f"   [OK] Model yuklendi (sinif sayisi: {detected_num_classes})")
    return model, detected_num_classes


def preprocess_image(image_path: Path, img_size: int = 512) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess image for inference."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img


def postprocess_mask(output: torch.Tensor, original_size: Tuple[int, int], num_classes: int) -> np.ndarray:
    """Postprocess model output to get segmentation mask."""
    # Handle different output formats
    if isinstance(output, dict):
        if "out" in output:
            output = output["out"]
        elif "logits" in output:
            output = output["logits"]
        else:
            output = list(output.values())[0]
    
    # Get predictions
    pred = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Resize to original size
    pred_resized = cv2.resize(pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred_resized


def create_colored_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Create colored visualization of segmentation mask."""
    # Create color map (different color for each class)
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        colored_mask[mask == class_id] = (colors[class_id][:3] * 255).astype(np.uint8)
    
    return colored_mask


def visualize_results(image: Image.Image, mask: np.ndarray, colored_mask: np.ndarray, 
                     output_path: Path, num_classes: int):
    """Visualize and save results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Orijinal Görüntü", fontsize=14)
    axes[0].axis("off")
    
    # Colored mask
    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation Mask (Renkli)", fontsize=14)
    axes[1].axis("off")
    
    # Overlay
    overlay = np.array(image) * 0.6 + colored_mask * 0.4
    overlay = overlay.astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Also save mask as PNG
    mask_path = output_path.parent / f"{output_path.stem}_mask.png"
    Image.fromarray(mask).save(mask_path)
    
    # Save colored mask
    colored_mask_path = output_path.parent / f"{output_path.stem}_colored_mask.png"
    Image.fromarray(colored_mask).save(colored_mask_path)


def infer_image(model: torch.nn.Module, image_path: Path, device: torch.device, 
                img_size: int, num_classes: int, use_amp: bool = True) -> Tuple[np.ndarray, Image.Image]:
    """Run inference on a single image."""
    # Preprocess
    img_tensor, original_img = preprocess_image(image_path, img_size)
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with autocast():
                output = model(img_tensor)
        else:
            output = model(img_tensor)
    
    # Handle tuple output (for models with aux_loss)
    if isinstance(output, dict):
        output = output["out"]  # Torchvision models return dict with "out" key
    elif isinstance(output, tuple):
        output = output[0]  # Use main output, ignore aux output
    
    # Postprocess
    mask = postprocess_mask(output, original_img.size, num_classes)
    
    return mask, original_img


def infer_model_name_from_checkpoint(checkpoint_path: Path) -> Tuple[str, int]:
    """Try to infer model name and num_classes from checkpoint or config."""
    # Try to find config file
    config_path = checkpoint_path.parent.parent / "configs" / "train_config.yaml"
    if not config_path.exists():
        config_path = Path("configs/train_config.yaml")
    
    model_name = "deeplabv3_resnet50"  # default
    num_classes = 2  # default
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if "model" in config:
                    model_name = config["model"].get("name", model_name)
                if "dataset" in config:
                    # Try to read classes.txt
                    classes_file = Path(config.get("dataset_root", "")) / "labels" / "segmentation" / "classes.txt"
                    if not classes_file.exists():
                        classes_file = Path(config.get("splits_dir", "")) / "train" / "masks" / "classes.txt"
                    if classes_file.exists():
                        with open(classes_file, "r", encoding="utf-8") as cf:
                            num_classes = len([line for line in cf if line.strip()])
        except Exception as e:
            print(f"[WARN] Config okunamadi: {e}, varsayilan degerler kullaniliyor")
    
    # Try to find classes.txt near model
    classes_candidates = [
        checkpoint_path.parent / "classes.txt",
        checkpoint_path.parent.parent / "splits" / "train" / "masks" / "classes.txt",
        Path("splits/train/masks/classes.txt"),
    ]
    for classes_file in classes_candidates:
        if classes_file.exists():
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    num_classes = len([line for line in f if line.strip()])
                print(f"[INFO] classes.txt bulundu: {num_classes} sinif")
                break
            except:
                pass
    
    return model_name, num_classes


def main():
    parser = argparse.ArgumentParser(description="Segmentation model inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--test_images", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Output directory for results")
    parser.add_argument("--model_name", type=str, default=None, 
                       help="Model name (deeplabv3_resnet50, fcn_resnet50, unet, etc.). Auto-detect if not provided.")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (including background). Auto-detect if not provided.")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], 
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Setup paths
    model_path = Path(args.model_path)
    test_images_dir = Path(args.test_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
    
    # Auto-detect model_name and num_classes if not provided
    if args.model_name is None or args.num_classes is None:
        inferred_model_name, inferred_num_classes = infer_model_name_from_checkpoint(model_path)
        if args.model_name is None:
            args.model_name = inferred_model_name
            print(f"[INFO] Model adi otomatik tespit edildi: {args.model_name}")
        if args.num_classes is None:
            args.num_classes = inferred_num_classes
            print(f"[INFO] Sinif sayisi otomatik tespit edildi: {args.num_classes}")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, actual_num_classes = load_model(model_path, args.model_name, args.num_classes, device)
    # Use actual num_classes from model
    args.num_classes = actual_num_classes
    
    # Find test images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    test_images = [f for f in test_images_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not test_images:
        raise ValueError(f"No images found in {test_images_dir}")
    
    print(f"\n[INFO] {len(test_images)} test goruntusu bulundu")
    print(f"[INFO] Sonuclar kaydedilecek: {output_dir}\n")
    
    # Run inference
    use_amp = AMP_AVAILABLE and device.type == "cuda"
    total_time = 0
    
    for idx, image_path in enumerate(test_images, 1):
        print(f"[{idx}/{len(test_images)}] İşleniyor: {image_path.name}")
        
        start_time = time.time()
        mask, original_img = infer_image(model, image_path, device, args.img_size, 
                                        args.num_classes, use_amp)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Create colored mask
        colored_mask = create_colored_mask(mask, args.num_classes)
        
        # Save results
        output_path = output_dir / f"{image_path.stem}_result.png"
        visualize_results(original_img, mask, colored_mask, output_path, args.num_classes)
        
        print(f"   [OK] Tamamlandi ({inference_time:.2f}s) -> {output_path.name}")
    
    avg_time = total_time / len(test_images)
    print(f"\n[OK] Tum goruntuler islendi!")
    print(f"   Toplam sure: {total_time:.2f}s")
    print(f"   Ortalama sure/goruntu: {avg_time:.2f}s")
    print(f"   Sonuclar: {output_dir}")


if __name__ == "__main__":
    main()

