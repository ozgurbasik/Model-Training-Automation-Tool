# TEDU Autonomous RC Car Model Training Tool

Otonom RC araçlar için nesne tespiti ve segmentasyon modelleri eğiten kapsamlı makine öğrenmesi platformu.

## 🚀 Özellikler

- **Çoklu Model Desteği**: YOLO, Faster R-CNN, DeepLabV3, UNet ve daha fazlası
- **Dataset Augmentation**: 15+ farklı augmentation tekniği ile dataset büyütme
- **MLflow Entegrasyonu**: Deney takibi ve model yönetimi
- **Streamlit Arayüzü**: Kullanıcı dostu web arayüzü
- **Çoklu Format Desteği**: LabelMe, Label Studio, YOLO, segmentation mask formatları
- **Otomatik Format Dönüşümü**: LabelMe ↔ Label Studio, Original → Merged format
- **Dataset Reorganizasyonu**: H1/H1_Annotations → images_all/labels_all dönüşümü

## 📋 Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch
- CUDA (GPU için önerilir)

### Adımlar

1. **Repo'yu klonla:**

```bash
cd RC-Car-Model-Training
```

2. **Sanal ortam oluştur:**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükle:**

```bash
pip install -r requirements.txt
```

## 🎯 Kullanım

### Streamlit Arayüzü (Önerilen)

```bash
streamlit run app/main.py
```

### Komut Satırı

```bash
python train.py --config configs/train_config.yaml
```

## 📁 Dataset Yapısı

Dataset'i repo'ya dahil değildir. 3 farklı dataset formatı desteklenir:

### Merged Format (Önerilen)

```
Dataset/
├── images_all/            # Tüm görüntüler (prefix ile)
│   ├── H1_frame_001.png
│   ├── H2_frame_001.png
│   └── ...
└── labels_all/            # Tüm anotasyonlar
    ├── H1_frame_001.json
    ├── H2_frame_001.json
    └── ...
```

### Original Format (Dönüştürülmesi Gerekir)

```
Dataset/
├── H1/                    # Görüntüler
│   ├── frame_001.png
│   └── ...
├── H1_Annotations/        # Anotasyonlar
│   ├── frame_001.json
│   └── ...
└── ...
```

**Not:** Original format'ı Streamlit arayüzünden "H1/H1_Annotations → images_all/labels_all Dönüşümü" butonu ile merged format'a çevirebilirsiniz.

## 🔄 Dataset İş Akışı

1. **Format Dönüşümü** (gerekirse):

   ```bash
   # Streamlit arayüzünden veya komut satırından
   python data/reorganize_to_merged.py
   ```

2. **Validation**:

   - Streamlit'te "Dataset Validation" sekmesini kullanın
   - LabelMe ve Label Studio formatları otomatik desteklenir

3. **Split**:

   - "Split Dataset" sekmesinde train/val/test oranlarını ayarlayın
   - Dosyalar otomatik olarak kopyalanır

4. **Eğitim**:
   - "Training" sekmesinden veya komut satırından eğitim başlatın

## ⚙️ Konfigürasyon

`configs/train_config.yaml` dosyasını düzenleyerek:

- Model parametreleri
- Augmentation ayarları
- Dataset path'leri
- Eğitim hiperparametreleri

## 🎨 Augmentation

Dataset büyütme için şu teknikler mevcut:

**Geometrik:**

- Yatay çevirme, rotasyon, ölçeklendirme
- Öteleme, kırpma

**Renk:**

- Parlaklık, kontrast, doygunluk
- Bulanıklaştırma, gürültü

**İleri Seviye:**

- Mixup, CutMix, Mosaic (detection)
- Elastik deformasyon (segmentation)

## 📊 MLflow

Deneyleri takip etmek için:

```bash
mlflow ui
```

## 🤖 Desteklenen Modeller

### Detection

- YOLOv8, YOLOv5
- Faster R-CNN, RetinaNet
- DETR, Cascade R-CNN

### Segmentation

- DeepLabV3, UNet
- SegFormer, Mask2Former
- PSPNet, HRNet

## 📝 Notlar

- Dataset boyutu nedeniyle repo'ya dahil edilmemiştir
- Model checkpoint'leri (.pt) repo'ya eklenmez
- MLflow verileri yerel olarak saklanır


## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır.
