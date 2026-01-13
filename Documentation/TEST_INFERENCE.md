# Segmentation Model Inference Kullanım Kılavuzu

Bu script, eğitilmiş segmentation modelini test görüntüleri üzerinde çalıştırır ve sonuçları görselleştirir.

## Kullanım

### Temel Kullanım

```bash
python test_segmentation.py \
    --model_path outputs/segmentation_last.pt \
    --test_images splits/test/images \
    --output_dir test_results
```

### Tüm Parametreler

```bash
python test_segmentation.py \
    --model_path outputs/segmentation_last.pt \
    --test_images splits/test/images \
    --output_dir test_results \
    --model_name deeplabv3_resnet50 \
    --num_classes 2 \
    --img_size 512 \
    --device auto
```

### Parametre Açıklamaları

- `--model_path`: Eğitilmiş model dosyasının yolu (zorunlu)
- `--test_images`: Test görüntülerinin bulunduğu klasör (zorunlu)
- `--output_dir`: Sonuçların kaydedileceği klasör (varsayılan: `test_results`)
- `--model_name`: Model adı (varsayılan: `deeplabv3_resnet50`)
  - Desteklenen modeller: `deeplabv3_resnet50`, `fcn_resnet50`, `unet`, `unet++`, `deeplabv3+`, `pspnet`, `fpn`, `linknet`, `segformer`, `hrnet`, `mask2former`
- `--num_classes`: Sınıf sayısı (varsayılan: 2)
- `--img_size`: Giriş görüntü boyutu (varsayılan: 512)
- `--device`: Kullanılacak cihaz (`auto`, `cuda`, `cpu`)

### Otomatik Tespit

Script, model adı ve sınıf sayısını otomatik olarak tespit edebilir:
- `configs/train_config.yaml` dosyasından model adını okur
- `classes.txt` dosyasından sınıf sayısını okur

Eğer bu dosyalar bulunamazsa, varsayılan değerler kullanılır.

## Çıktılar

Script her test görüntüsü için 3 dosya oluşturur:

1. `{image_name}_result.png`: Orijinal görüntü, renkli mask ve overlay'i gösteren görselleştirme
2. `{image_name}_mask.png`: Siyah-beyaz segmentation mask
3. `{image_name}_colored_mask.png`: Renkli segmentation mask

## Örnek

```bash
# Basit kullanım
python test_segmentation.py \
    --model_path D:\RcCarModelTraining\outputs\segmentation_last.pt \
    --test_images D:\RcCarModelTraining\splits\test\images

# Tüm parametrelerle
python test_segmentation.py \
    --model_path D:\RcCarModelTraining\outputs\segmentation_last.pt \
    --test_images D:\RcCarModelTraining\splits\test\images \
    --output_dir test_results \
    --model_name deeplabv3_resnet50 \
    --num_classes 2 \
    --img_size 512
```

## Notlar

- Model ve görüntüler otomatik olarak GPU'ya yüklenir (varsa)
- Mixed precision inference kullanılır (GPU varsa)
- Her görüntü için inference süresi gösterilir
- Sonuçlar `test_results` klasörüne kaydedilir

