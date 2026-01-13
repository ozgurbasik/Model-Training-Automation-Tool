import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- configure paths ---
MODEL_PATH = r"C:\Users\basik\Desktop\OzgurLocal\RC_CV\RC-Car-Model-Training\Yolov8-segİlk4.01.26.pt"  # <-- set your YOLOv8-seg weights
TEST_DIR = r"C:\Users\basik\Desktop\OzgurLocal\RC_CV\RC-Car-Model-Training\splits\test\images"
OUT_DIR = r"C:\Users\basik\Desktop\OzgurLocal\RC_CV\RC-Car-Model-Training\predict_yolov8"
IMG_SIZE = 640
ALPHA = 0.4  # overlay transparency (0..1)

# Set your class names (optional) and colors (BGR)
CLASS_NAMES = ["background", "class1", "class2", "class3", "class4"]
CLASS_COLORS = np.array([
    [0, 0, 0],        # background
    [0, 0, 255],      # class 1 - red
    [0, 255, 0],      # class 2 - green
    [255, 0, 0],      # class 3 - blue
    [0, 255, 255],    # class 4 - yellow
    [255, 0, 255],    # extra colors if more classes
    [255, 255, 0],
], dtype=np.uint8)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    model.to(device)

    for name in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, name)
        original_bgr = cv2.imread(img_path)
        if original_bgr is None:
            print(f"Skip unreadable file: {name}")
            continue

        h, w = original_bgr.shape[:2]
        results = model(img_path, imgsz=IMG_SIZE, device=device, verbose=False)[0]

        if results.masks is None or results.boxes is None or len(results.masks.data) == 0:
            cv2.imwrite(os.path.join(OUT_DIR, f"overlay_{name}"), original_bgr)
            continue

        masks = results.masks.data.cpu().numpy()  # (N, Hm, Wm)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        composite_mask = np.zeros((h, w), dtype=np.uint8)

        for mask, cls_id in zip(masks, cls_ids):
            resized = cv2.resize(mask, (w, h))
            bin_mask = (resized > 0.5).astype(np.uint8)
            composite_mask[bin_mask == 1] = (cls_id + 1)  # background=0, classes start at 1

        # Save grayscale composite mask scaled to 0-255
        max_cls = max(1, composite_mask.max())
        mask_uint8 = (composite_mask.astype(float) / max_cls * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, f"mask_{name}"), mask_uint8)

        # Colorize per class then blend once over original
        colors = CLASS_COLORS[: len(CLASS_COLORS)]
        colored_mask = colors[composite_mask.clip(0, len(colors) - 1)]  # HxWx3 BGR
        overlay = cv2.addWeighted(original_bgr, 0.6, colored_mask, 0.4, 0)
        cv2.imwrite(os.path.join(OUT_DIR, f"overlay_{name}"), overlay)

        # Optional: print detected classes per image
        unique_ids = sorted(set(int(x) for x in cls_ids))
        labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"cls{i}" for i in unique_ids]
        print(f"{name}: classes -> {labels}")


if __name__ == "__main__":
    main()
