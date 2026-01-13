# MLflow Kullanımı - RC Car Model Training Projesi

Bu projede **MLflow**, makine öğrenmesi deneylerini takip etmek, metrikleri kaydetmek ve modelleri yönetmek için kullanılmaktadır.

## MLflow'un Kullanım Amaçları

### 1. **Deney Takibi (Experiment Tracking)**
- Her eğitim çalışması (run) MLflow tarafından otomatik olarak kaydedilir
- Deney adı: `"autonomous_vehicle"`
- Tracking URI: `"mlruns"` (yerel dizin)

### 2. **Parametre Kaydı (Parameter Logging)**
Eğitim sırasında şu parametreler kaydedilir:
- `task_type`: Görev tipi (detection veya segmentation)
- `common_epochs`: Epoch sayısı
- `common_batch_size`: Batch size
- `common_lr`: Learning rate
- `common_weight_decay`: Weight decay
- `detection_model_name` veya `segmentation_model_name`: Kullanılan model adı

### 3. **Metrik Takibi (Metrics Tracking)**

#### Detection Görevleri İçin:
- **`train_loss_detection`**: Her epoch için eğitim loss değeri
- **`val_loss_detection`**: Her epoch için validation loss değeri
- **`precision_50_detection`**: IoU threshold 0.5 için precision
- **`recall_50_detection`**: IoU threshold 0.5 için recall
- **`f1_50_detection`**: IoU threshold 0.5 için F1 score

#### Segmentation Görevleri İçin:
- **`train_loss_segmentation`**: Her epoch için eğitim loss değeri
- **`val_loss_segmentation`**: Her epoch için validation loss değeri
- **`pixel_acc_segmentation`**: Pixel accuracy
- **`miou_segmentation`**: Mean Intersection over Union (mIoU)

### 4. **Model Kaydı (Model Registry)**
- Eğitilen modeller `mlflow.pytorch.log_model()` ile kaydedilir
- Model checkpoint'leri `mlflow.log_artifact()` ile kaydedilir
- Bu sayede modeller versiyonlanabilir ve geri yüklenebilir

### 5. **Streamlit Dashboard Entegrasyonu**
`app/main.py` dosyasında MLflow kullanılarak:
- Tüm deneyler listelenir
- Farklı run'ların metrikleri karşılaştırılır
- Loss grafikleri görselleştirilir
- Detection ve segmentation metrikleri ayrı ayrı gösterilir

## Kod Örnekleri

### Eğitim Sırasında MLflow Kullanımı (`train.py`)

```python
# MLflow başlatma
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("autonomous_vehicle")

with mlflow.start_run():
    # Parametreleri kaydet
    mlflow.log_param("task_type", task_type)
    mlflow.log_param("common_epochs", epochs)
    
    # Metrikleri kaydet
    mlflow.log_metric("train_loss_detection", avg_loss, step=epoch)
    mlflow.log_metric("val_loss_detection", avg_val_loss, step=epoch)
    
    # Modeli kaydet
    mlflow.pytorch.log_model(model, artifact_path="detection_model")
```

### Dashboard'da MLflow Kullanımı (`app/main.py`)

```python
# MLflow client oluşturma
mlflow.set_tracking_uri("mlruns")
client = mlflow.tracking.MlflowClient()

# Deneyleri sorgulama
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"params.task_type = '{task_type}'"
)

# Metrik geçmişini alma
hist = client.get_metric_history(run_id, "train_loss_detection")
```

## Avantajlar

1. **Deney Karşılaştırması**: Farklı hiperparametrelerle yapılan deneyler kolayca karşılaştırılabilir
2. **Model Versiyonlama**: Her model versiyonu kaydedilir ve geri yüklenebilir
3. **Reproducibility**: Tüm parametreler ve sonuçlar kaydedildiği için deneyler tekrarlanabilir
4. **Görselleştirme**: Streamlit dashboard ile metrikler görsel olarak incelenebilir
5. **Organizasyon**: Tüm deneyler merkezi bir yerde toplanır

## Dosya Yapısı

MLflow verileri `mlruns/` dizininde saklanır:
```
mlruns/
├── 0/
│   └── meta.yaml  # Experiment metadata
└── [experiment_id]/
    └── [run_id]/
        ├── metrics/  # Metrikler
        ├── params/   # Parametreler
        └── artifacts/ # Modeller ve diğer dosyalar
```

## Özet

MLflow bu projede, RC Car model eğitimi sürecinde:
- ✅ Deneyleri organize etmek
- ✅ Metrikleri takip etmek
- ✅ Modelleri kaydetmek ve versiyonlamak
- ✅ Streamlit dashboard'da sonuçları görselleştirmek

için kullanılmaktadır.




