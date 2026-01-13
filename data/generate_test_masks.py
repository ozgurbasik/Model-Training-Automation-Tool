"""
Quick script to generate PNG masks from test split JSON files.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Load classes
classes_file = Path("splits_seg/train/labels/classes.txt")
if not classes_file.exists():
    classes_file = Path("splits_seg/val/labels/classes.txt")
label_map = {}
if classes_file.exists():
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                label_map[parts[1]] = int(parts[0])

print(f"Loaded {len(label_map)} classes: {label_map}")

# Convert test JSONs to PNGs
json_dir = Path("splits_seg/test/labels")
image_dir = Path("splits_seg/test/images")
output_dir = Path("splits_seg/test/labels")
output_dir.mkdir(parents=True, exist_ok=True)

count = 0
for json_path in sorted(json_dir.glob("*.json")):
    with open(json_path, 'r') as f:
        anno = json.load(f)
    
    # Get image dimensions
    img_path = image_dir / f"{json_path.stem}.png"
    if not img_path.exists():
        img_path = image_dir / f"{json_path.stem}.jpg"
    
    if img_path.exists():
        img = Image.open(img_path)
        w, h = img.size
    else:
        # Fallback: use dimensions from JSON
        h = anno.get("imageHeight", 640)
        w = anno.get("imageWidth", 640)
    
    # Create blank mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw shapes
    shapes = anno.get("shapes", [])
    # Also check for Label Studio format
    if not shapes:
        shapes = anno.get("labels", [])
    
    for shape in shapes:
        # Try LabelMe format first
        label = shape.get("label", "")
        # Try Label Studio format
        if not label:
            polygon_labels = shape.get("polygonlabels", [])
            if polygon_labels:
                label = polygon_labels[0]
        
        if not label or label not in label_map:
            continue
        
        class_id = label_map[label]
        points = shape.get("points", [])
        
        if len(points) < 3:
            continue
        
        # Convert to PIL format
        polygon_points = [(int(x), int(y)) for x, y in points]
        
        # Draw filled polygon
        mask_img = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(polygon_points, fill=class_id)
        mask = np.array(mask_img)
    
    # Save as PNG
    out_path = output_dir / f"{json_path.stem}.png"
    Image.fromarray(mask).save(out_path)
    count += 1
    print(f"✅ {out_path.name}")

print(f"\n✅ Generated {count} PNG masks in {output_dir}")
