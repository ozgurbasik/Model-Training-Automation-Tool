# Dataset Format Dokümantasyonu

## Desteklenen Dataset Yapıları

Bu proje 3 farklı dataset formatını destekler:

### 1. Merged Format (Önerilen)

```
Dataset/
├── images_all/                  # Tüm görüntüler tek klasörde
│   ├── H1_frame_001.png        # Prefix ile ayırt edilir
│   ├── H1_frame_002.png
│   ├── H2_frame_001.png
│   └── ...
└── labels_all/                  # Tüm anotasyonlar tek klasörde
    ├── H1_frame_001.json
    ├── H1_frame_002.json
    ├── H2_frame_001.json
    └── ...
```

**Avantajları:**

- Daha hızlı dosya erişimi
- Split işlemleri için ideal
- Eğitim pipeline'ı tarafından desteklenir

### 2. Reorganized Format

```
Dataset/
├── images/                      # Tüm görüntüler
│   ├── H1_frame_001.png
│   ├── H2_frame_001.png
│   └── ...
└── labels/                      # Tüm anotasyonlar
    ├── H1_frame_001.json
    ├── H2_frame_001.json
    └── ...
```

### 3. Original Format (Eski Format)

```
Dataset/
├── H1/                          # Görüntü klasörü
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
├── H1_Annotations/              # LabelMe JSON anotasyonları
│   ├── frame_001.json
│   ├── frame_002.json
│   └── ...
├── H2/
│   └── ...
├── H2_Annotations/
│   └── ...
└── ...
```

**Not:** Bu format artık eğitim ve split işlemleri için desteklenmemektedir. Merged veya Reorganized formata dönüştürülmesi gerekir.

### Format Dönüşümü

Streamlit arayüzünde **Dataset Validation** sekmesinden:

1. **Original → Merged Format:**

   - "H1/H1_Annotations → images_all/labels_all Dönüşümü" butonunu kullanın
   - Dosyalar otomatik olarak prefix ile yeniden adlandırılır
   - Orijinal klasörler silinir

2. **LabelMe → Label Studio:**
   - "LabelMe → Label Studio Formatına Çevir" butonunu kullanın
   - Koordinatlar piksel tabanlı → yüzde tabanlı olarak dönüştürülür

### Önemli Kurallar:

1. **Dosya Eşleştirme:**

   - Her JSON dosyası aynı isimde bir görüntü dosyasına karşılık gelmelidir
   - Örnek: `H1_frame_001.json` → `H1_frame_001.png` veya `H1_frame_001.jpg`

2. **Split ve Eğitim:**
   - Sadece **Merged** veya **Reorganized** format desteklenir
   - Original format kullanıyorsanız önce dönüştürün

## Anotasyon Formatları

### LabelMe JSON Formatı

Her JSON dosyası şu yapıda olmalıdır:

```json
{
  "version": "4.5.7",
  "flags": {},
  "shapes": [
    {
      "label": "Human",
      "points": [
        [100.5, 200.3],
        [150.2, 200.1],
        [150.0, 250.5],
        [100.0, 250.0]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "frame_001.png", // Opsiyonel: yoksa dosya adından çıkarılır
  "imageData": null,
  "imageHeight": 720,
  "imageWidth": 1280
}
```

### Label Studio JSON Formatı

Label Studio formatı da desteklenmektedir (koordinatlar yüzde tabanlı):

```json
{
  "version": "4.5.7",
  "flags": {},
  "labels": [
    {
      "label": "Human",
      "points": [
        [7.85, 27.82],
        [11.73, 27.79],
        [11.72, 34.79],
        [7.81, 34.72]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "frame_001.png",
  "imageData": null,
  "imageHeight": 720,
  "imageWidth": 1280
}
```

**Farklar:**

- LabelMe: `shapes` alanı, piksel koordinatları (örn: [100.5, 200.3])
- Label Studio: `labels` alanı, yüzde koordinatları (örn: [7.85, 27.82])

**Not:** Validation sistemi Label Studio formatını otomatik olarak algılar ve LabelMe formatına çevirir (sadece doğrulama için, dosyalar değişmez).

### LabelMe JSON Alanları:

- **`shapes`** (opsiyonel): Anotasyon listesi

  - Boş olabilir (sadece uyarı verir, hata değil)
  - Her shape:
    - `label`: Sınıf adı (örn: "Human", "Car", "Cone")
    - `points`: Poligon noktaları listesi `[[x1, y1], [x2, y2], ...]`
    - En az 3 nokta olmalı (poligon için)

- **`imagePath`** (opsiyonel): Görüntü dosya adı
  - Yoksa: JSON dosya adından çıkarılır (örn: `frame_001.json` → `frame_001.png`)

## Detection (YOLO) Formatı

Eğer `--check-detection` kullanılırsa, şu yapı beklenir:

```
labels/detection/
├── H1/                          # Split adı (H1_Annotations → H1)
│   ├── frame_001.txt
│   ├── frame_002.txt
│   └── ...
├── H2/
│   └── ...
└── classes.txt                  # Sınıf listesi
```

### YOLO Label Formatı:

Her `.txt` dosyası şu formatta olmalıdır:

```
class_id center_x center_y width height
```

- Tüm değerler **normalize edilmiş** (0-1 arası)
- `center_x, center_y`: Bounding box merkezi
- `width, height`: Bounding box genişlik ve yüksekliği
- Her satır bir nesne için

**Örnek:**

```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

### classes.txt Formatı:

```
0 Human
1 Car
2 Cone
```

## Segmentation Formatı

Eğer `--check-segmentation` kullanılırsa, şu yapı beklenir:

```
labels/segmentation/
├── H1/                          # Split adı
│   ├── frame_001.png            # Mask dosyası
│   ├── frame_002.png
│   └── ...
├── H2/
│   └── ...
└── classes.txt                  # Sınıf listesi
```

### Segmentation Mask Formatı:

- Her mask dosyası görüntü ile **aynı boyutta** olmalıdır
- Her piksel bir **sınıf ID'si** içerir (0 = background)
- Grayscale PNG formatında

### classes.txt Formatı:

```
0 background
1 Human
2 Car
3 Cone
```

## Validation Kontrolleri

### Format Desteği:

✅ **Otomatik Format Algılama:**

- LabelMe formatı (shapes alanı)
- Label Studio formatı (labels alanı)
- Label Studio dosyaları otomatik olarak LabelMe'ye çevrilir (sadece doğrulama için)

✅ **Desteklenen Dataset Yapıları:**

- Merged format (images_all/labels_all)
- Reorganized format (images/labels)
- Original format (sadece validation için, split/train için desteklenmez)

### LabelMe/Label Studio Format Kontrolleri:

✅ **Geçerli:**

- JSON formatı geçerli
- Görüntü dosyası bulunuyor
- Anotasyonlar varsa, format doğru
- Label Studio formatı otomatik dönüştürülür

⚠️ **Uyarılar (Hata Değil):**

- `imagePath` alanı yok (dosya adından çıkarılır)
- `shapes`/`labels` alanı boş (anotasyon yok ama format doğru)
- Noktalar görüntü sınırları dışında (küçük farklar tolere edilir)

❌ **Hatalar:**

- JSON formatı geçersiz
- Görüntü dosyası bulunamıyor
- Shape/label'lerde eksik `label` veya `points`
- Noktalar geçersiz format

### Detection Format Kontrolleri:

✅ **Geçerli:**

- Label dosyası var
- Format doğru (5 değer: class_id cx cy w h)
- Koordinatlar normalize (0-1 arası)
- Bounding box geçerli (width/height > 0)

❌ **Hatalar:**

- Label dosyası yok
- Format yanlış (5 değer değil)
- Koordinatlar normalize değil
- Geçersiz bounding box

### Segmentation Format Kontrolleri:

✅ **Geçerli:**

- Mask dosyası var
- Mask boyutu görüntü boyutuyla eşleşiyor
- Mask değerleri geçerli

⚠️ **Uyarılar:**

- Mask tamamen boş (tüm pikseller 0)

❌ **Hatalar:**

- Mask dosyası yok
- Boyut uyuşmazlığı
- Mask okunamıyor

## Kullanım Örnekleri

### Komut Satırından:

```bash
# Sadece LabelMe formatını kontrol et
python data/validate_dataset.py --dataset-root Dataset --check-labelme

# Detection formatını da kontrol et
python data/validate_dataset.py \
    --dataset-root Dataset \
    --check-labelme \
    --check-detection \
    --detection-labels-root labels/detection

# Tüm formatları kontrol et
python data/validate_dataset.py \
    --dataset-root Dataset \
    --check-labelme \
    --check-detection \
    --check-segmentation \
    --detection-labels-root labels/detection \
    --segmentation-masks-root labels/segmentation
```

### Streamlit Arayüzünden:

**Önerilen İş Akışı:**

1. Streamlit'i başlat: `streamlit run app/main.py`
2. "Dataset Validation" sekmesine git
3. Dataset kök dizinini gir (örn: `DataSet/RcCArDataset`)

4. **Format Dönüşümü (gerekirse):**

   - Original format kullanıyorsanız: "H1/H1_Annotations → images_all/labels_all Dönüşümü" butonuna tıklayın
   - LabelMe formatını Label Studio'ya çevirmek için: "LabelMe → Label Studio Formatına Çevir" butonunu kullanın

5. **Validation:**

   - Kontrol edilecek formatları seç
   - "Dataset'i Kontrol Et" butonuna tıkla
   - Label Studio formatı otomatik algılanır ve doğrulanır

6. **Split:**
   - "Split Dataset" sekmesine git
   - Dataset yolunu girin (merged veya reorganized format gerekir)
   - Train/val/test oranlarını ayarlayın
   - "Dataset'i Böl" butonuna tıklayın

## Örnek Dataset Yapısı

```
Dataset/
├── H1/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── frame_003.png
├── H1_Annotations/
│   ├── frame_001.json  (Human, Car anotasyonları içerir)
│   ├── frame_002.json  (boş - sadece uyarı verir)
│   └── frame_003.json  (Cone anotasyonları içerir)
├── H2/
│   └── ...
└── H2_Annotations/
    └── ...
```

## Notlar

- Görüntü formatları: `.png`, `.jpg`, `.jpeg` (büyük/küçük harf duyarsız)
- JSON dosyaları UTF-8 encoding ile olmalı
- Boş anotasyonlar (shapes yok) geçerli kabul edilir, sadece uyarı verilir
- Path sorunları otomatik olarak çözülmeye çalışılır (relative path'ler, farklı uzantılar)
