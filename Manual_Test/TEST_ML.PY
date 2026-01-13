import os, cv2, torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 eğitimde ne kullandıysan AYNISI
num_classes = 4   # değiştir (örn: 3) eğer multi-class ise - checkpoint 4 sınıf ile eğitilmiş

# Checkpoint şu yapıya sahip: backbone, classifier, aux_classifier (DeepLabV3 tarzı)
# Eğer smp.Unet istiyorsan, eğitimde kullanılan model mimarisini kontrol et
# Bu örnekte DeepLabV3 varsayılıyor; Unet istiyorsan aşağıdaki satırları kapat
try:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(r"C:\Users\basik\Desktop\OzgurLocal\RC_CV\RC-Car-Model-Training\outputs\segmentation_best_Testing_for_ozgur.pt", map_location=device, weights_only=False))
except RuntimeError as e:
    print(f"⚠️ smp.Unet ile yükleme başarısız: {e}")
    print("DeepLabV3 deneniyor...")
    
    # DeepLabV3 deneme
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(r"C:\Users\basik\Desktop\OzgurLocal\RC_CV\RC-Car-Model-Training\outputs\segmentation_best_Testing_for_ozgur.pt", map_location=device, weights_only=False))
        print("✅ DeepLabV3 ile başarıyla yüklendi")
    except RuntimeError as e2:
        print(f"❌ DeepLabV3 ile de başarısız: {e2}")
        print("Lütfen eğitimde kullanılan model mimarisini kontrol edin ve aşağıda ayarlayın")
        raise
    
    model = model.to(device)
model.eval()

test_dir = r"C:\Users\basik\Desktop\OzgurLocal\RC_CV\RC-Car-Model-Training\splits\test\images"
out_dir = "predict"
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    for name in os.listdir(test_dir):
        img = cv2.imread(os.path.join(test_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = img / 255.0

        img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

        pred = model(img)

        if num_classes == 1:
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            mask = pred[0,0].cpu().numpy()
        else:
            pred = torch.softmax(pred, dim=1)
            mask = torch.argmax(pred, dim=1)[0].cpu().numpy()

        # Save grayscale mask (0..num_classes-1 scaled to 0-255)
        mask_uint8 = (mask.astype(float) / max(num_classes - 1, 1) * 255).astype("uint8")
        cv2.imwrite(os.path.join(out_dir, f"mask_{name}"), mask_uint8)

        # Define distinct BGR colors per class (extendable)
        class_colors = np.array([
            [0, 0, 0],        # background
            [0, 0, 255],      # class 1 - red
            [0, 255, 0],      # class 2 - green
            [255, 0, 0],      # class 3 - blue
            [0, 255, 255],    # class 4 - yellow (if exists)
        ], dtype=np.uint8)
        colors = class_colors[:max(num_classes, 1)]
        colored_mask = colors[mask.clip(0, len(colors)-1)]  # HxWx3 BGR

        # Load original in BGR and resize to 512x512 to match mask
        original_bgr = cv2.imread(os.path.join(test_dir, name))
        original_bgr = cv2.resize(original_bgr, (512, 512))

        # Blend: 60% original, 40% mask
        overlay = cv2.addWeighted(original_bgr, 0.6, colored_mask, 0.4, 0)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{name}"), overlay)