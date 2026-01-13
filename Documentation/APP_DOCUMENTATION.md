# Autonomous Vehicle Training Dashboard - Application Documentation

## Overview

This is a **Streamlit-based web application** designed for training autonomous vehicle models with support for **Object Detection** and **Segmentation** tasks. The app provides a complete workflow for dataset management, training, and experiment monitoring.

---

## Main Interface

### Top Level Controls

- **Task Type Selector** (Left Sidebar)
  - Dropdown menu: "Object Detection" or "Segmentation"
  - Determines which models and configurations are available throughout the app
  - Changes are applied globally across all tabs

---

## Tabs Overview

The application is organized into **5 main tabs**:

1. **Dataset Import** - Download datasets from Hugging Face
2. **Dataset Validation** - Check and convert dataset formats
3. **Dataset Split** - Divide dataset into train/val/test sets
4. **Train** - Configure and run model training
5. **Experiments** - Monitor training progress and manage experiments

---

## Tab 1: Dataset Import

### Purpose

Download datasets from Hugging Face Hub to your local machine.

### Controls & Buttons

#### **Repo ID** (Text Input)

- **Purpose**: Specify the Hugging Face dataset repository
- **Default Value**: `TargetU/RcCArDataset`
- **Example**: `username/dataset-name`

#### **Yerel dizin adı** (Local Directory) (Text Input)

- **Purpose**: Set where the dataset will be saved locally
- **Default Value**: `./DataSet/RcCArDataset`
- **Recommendation**: Save all datasets in the `DataSet/` folder
- **Example**: `./DataSet/MyDataset`

#### **Sembolik link kullan** (Use Symlinks) (Checkbox)

- **Purpose**: Enable symbolic links if available (faster, saves disk space)
- **Default**: Unchecked (disabled)
- **Use When**: You want to save storage space and have filesystem support

#### **Dataseti indir** (Download Dataset) (Primary Button)

- **Function**:
  - Validates the Repo ID
  - Downloads the dataset from Hugging Face to the specified directory
  - Shows progress and success/error messages
- **Success**: Displays "✅ İndirildi: [path]"
- **Error**: Displays error message with details

### Workflow

1. Enter repository ID (or use default)
2. Set local directory path
3. Optionally enable symlinks
4. Click "Dataseti indir" button
5. Wait for download to complete

---

## Tab 2: Dataset Validation

### Purpose

Validate dataset formats, convert between formats, reorganize dataset structure, and check for data quality issues.

### Controls & Buttons

#### **Dataset kök dizini** (Dataset Root Directory) (Text Input)

- **Purpose**: Path to the dataset folder
- **Default**: `Dataset`
- **Required Format**: Full path (e.g., `C:/Users/.../RC-Car-Model-Training/DataSet/RcCArDataset`)

#### **Dönüştürürken .backup oluştur** (Create Backup During Conversion) (Checkbox)

- **Purpose**: Automatically backup original files before conversion
- **Default**: Checked (enabled)
- **Recommended**: Keep enabled

---

### Conversion Buttons

#### **LabelMe -> Label Studio'ya çevir** (Convert LabelMe to Label Studio)

- **Function**:
  - Converts LabelMe JSON format to Label Studio format
  - Automatically creates backups (if enabled)
  - Updates polygon coordinates
- **Output**:
  - Converted file count
  - Total files processed
  - Error count with details
  - Sample preview of converted image with annotations
- **Use When**: Dataset is in LabelMe format but you need Label Studio format

#### **H1/H1_Annotations → images_all/labels_all Dönüşümü** (Reorganize to Merged Format)

- **Function**:
  - Converts H1, H1_Annotations folder structure to unified format
  - Creates `images_all/` and `labels_all/` directories
  - Adds prefixes to filenames (e.g., `H1_frame_001.png`)
  - **Deletes** original folders after conversion
- **Output**: Success message and reorganization details
- **Warning**: ⚠️ Original folders are deleted - ensure you have backups!
- **Use When**: You need to consolidate data from multiple source folders

#### **Dikey (Vertical) Resimleri Yatay (Horizontal) Yap** (Rotate Vertical Images)

- **Function**:
  - Finds and rotates vertical (portrait) images to horizontal (landscape)
  - Automatically updates JSON annotations with new coordinates
  - Skips already-horizontal images
- **Output**:
  - Count of rotated images
  - Total vertical images found
  - Error details if any
  - Sample preview showing rotated image with updated annotations
- **Use When**: Dataset contains mixed orientation images

---

### Validation Checkboxes

#### **LabelMe & Label Studio Formatını Kontrol Et** (Check LabelMe & Label Studio Format)

- **Default**: Checked
- **Purpose**: Validate annotation format compatibility

#### **Detection (YOLO) formatını kontrol et** (Check Detection YOLO Format)

- **Default**: Unchecked
- **Enable to**: Also validate YOLO detection format files
- **Shows**: Detection labels directory input when enabled

#### **Segmentation formatını kontrol et** (Check Segmentation Format)

- **Default**: Unchecked
- **Enable to**: Also validate segmentation mask format
- **Shows**: Segmentation masks directory input when enabled

---

### Validation Execution

#### **Dataset'i Kontrol Et** (Check Dataset)

- **Function**:
  - Scans dataset directory
  - Validates all selected formats
  - Checks for missing labels or images
  - Generates quality statistics
- **Output**:
  - Summary statistics (total images, labels, mismatches)
  - Format-specific validation results
  - Error details and warnings
  - List of problematic files
  - Visualization of sample images with annotations

---

## Tab 3: Dataset Split

### Purpose

Divide the dataset into training, validation, and test sets with customizable ratios.

### Controls & Buttons

#### **Dataset kök dizini** (Dataset Root Directory) (Text Input)

- **Purpose**: Path to dataset to split
- **Default**: `Dataset_Merged`
- **Required Format**: Full path

#### **Train oranı** (Train Ratio) (Slider)

- **Range**: 0.0 to 1.0 (0% to 100%)
- **Default**: 0.7 (70%)
- **Step**: 0.05 (5% increments)

#### **Validation oranı** (Validation Ratio) (Slider)

- **Range**: 0.0 to 1.0
- **Default**: 0.2 (20%)
- **Step**: 0.05

#### **Test oranı** (Test Ratio) (Slider)

- **Range**: 0.0 to 1.0
- **Default**: 0.1 (10%)
- **Step**: 0.05

#### **Ratio Validation Display**

- **Warning Message**: Shows if ratios don't sum to 1.0
- **Success Message**: Shows ✓ and total sum when valid (e.g., 1.00)
- **Must Sum To**: Approximately 1.0 (within 0.01 tolerance)

#### **Çıktı dizini** (Output Directory) (Text Input)

- **Purpose**: Where to save split files
- **Default**: `splits`
- **Creates**: `train.txt`, `val.txt`, `test.txt` files

#### **Random seed** (Number Input)

- **Purpose**: Ensures reproducible splits
- **Default**: 42
- **Range**: 0 to 9999
- **Tip**: Use same seed for consistent splits across runs

#### **Dosyaları train/val/test klasörlerinde images klasörü oluştur ve kopyala** (Copy Files to Split Folders) (Checkbox)

- **Default**: Unchecked
- **When Enabled**:
  - Creates `train/images/`, `train/labels/`, `val/images/`, etc.
  - Copies image and label files to respective folders
  - Doubles disk space usage
- **When Disabled**: Only creates `.txt` files with relative paths

---

### Split Execution

#### **Dataset'i Böl** (Split Dataset)

- **Function**:
  - Validates ratios sum to 1.0
  - Splits images and labels into train/val/test
  - Creates split `.txt` files
  - Optionally copies files to organized folders
- **Validations**:
  - Dataset directory must exist
  - Ratios must sum to 1.0
  - Images and labels are paired
- **Output**:
  - Success message with split statistics
  - Metrics showing count and percentage for each set
  - List of created files

---

## Tab 4: Train

### Purpose

Configure model hyperparameters and train object detection or segmentation models.

### Main Controls

#### **Splits dizini** (Splits Directory) (Text Input)

- **Purpose**: Directory containing `train.txt`, `val.txt`, `test.txt`
- **Default**: `splits`
- **Required**: Yes
- **Automatically**: Script finds images and labels from these split files

#### **Advanced Settings** (Expandable Section)

##### **Detection Labels Directory** (Optional, shown when task = detection)

- **Purpose**: Path to detection labels if auto-detection fails
- **Default**: Auto-detect

##### **Segmentation Masks Directory** (Optional, shown when task = segmentation)

- **Purpose**: Path to segmentation masks if auto-detection fails
- **Default**: Auto-detect

---

### Model & Hyperparameter Configuration

#### **Deney adı** (Experiment Name) (Text Input)

- **Purpose**: Custom name for this training run
- **Default**: Empty (auto-generated from config)
- **Example**: `deeplabv3_batch4_lr0.0005`
- **Tip**: Use descriptive names for easy identification

#### **Detection modeli** (Detection Model) (Dropdown - shown when task = detection)

- **Available Models**:
  - YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
  - YOLOv5: `yolov5n`, `yolov5s`, `yolov5m`, `yolov5l`, `yolov5x`
  - Others: `detr`, `cascade_rcnn`, `efficientdet_d0-d2`, `fcos`, `atss`, `fasterrcnn`, `retinanet`, `ssd300_vgg16`
- **Default**: (First in list)
- **Recommendation**: Start with `yolov8s` for balance

#### **Segmentation modeli** (Segmentation Model) (Dropdown - shown when task = segmentation)

- **Available Models**:
  - Advanced: `mask2former`, `segnext`, `bisenetv2`, `ddrnet`, `pidnet`, `topformer`, `segformer`
  - Standard: `deeplabv3+`, `pspnet`, `hrnet`, `deeplabv3_resnet50`, `fcn_resnet50`
  - Lightweight: `unet`, `unet++`
- **Default**: (First in list)
- **Recommendation**: Start with `segformer` for balance

#### **Epochs** (Number Input)

- **Purpose**: Number of training epochs
- **Range**: 1 to 200
- **Default**: 10
- **Tip**: Start with 10-20 for testing, increase for production

#### **Batch size** (Number Input)

- **Purpose**: Number of images per batch
- **Range**: 1 to 64
- **Default**: 4
- **Note**: Higher = faster but requires more GPU memory

#### **Learning rate** (Number Input)

- **Purpose**: Learning rate for optimizer
- **Range**: 1e-6 to 1e-1
- **Default**: 5e-4 (0.0005)
- **Format**: Scientific notation
- **Tip**: Start with default, reduce if loss doesn't decrease

#### **Eğitim sonrası gösterilecek görsel sayısı** (Preview Samples) (Slider)

- **Purpose**: How many validation images to display after training
- **Range**: 0 to 10
- **Default**: (Based on config)
- **Note**: More previews = longer load time

---

### Augmentation Settings (Expandable Section)

#### **Veri Artırma Aç** (Enable Augmentation) (Checkbox)

- **Default**: Unchecked
- **Purpose**: Apply data augmentation to training set

#### **Validation İçin De Artırma Yap** (Augment Validation) (Checkbox)

- **Default**: Unchecked
- **Purpose**: Also augment validation data

#### **Augmentation Parameters** (Sliders - shown when augmentation enabled)

All values are probabilities (0.0 to 1.0) or strength (magnitude):

- **horizontal_flip**: Probability of horizontal flip (default: 0.5)
- **rotation**: Rotation angle in degrees (default: 10)
- **scale**: Scale factor (default: 0.1)
- **translation**: Translation amount (default: 0.1)
- **crop_scale**: Crop scale factor (default: 0.8)
- **brightness**: Brightness change (default: 0.2)
- **contrast**: Contrast change (default: 0.2)
- **saturation**: Saturation change (default: 0.2)
- **hue**: Hue shift (default: 0.1)
- **blur**: Blur intensity (default: 0.1)
- **noise**: Noise intensity (default: 0.1)
- **mixup**: MixUp probability (default: 0.0)
- **cutmix**: CutMix probability (default: 0.0)
- **mosaic**: Mosaic probability (default: 0.0)
- **elastic**: Elastic deformation (default: 0.1)
- **grid_distortion**: Grid distortion (default: 0.1)

---

### Training Execution

#### **Eğitim Başlat** (Start Training) (Primary Button)

- **Function**:
  - Writes configuration to `configs/train_config.yaml`
  - Launches training process
  - Displays real-time training output
  - Shows validation previews upon completion
- **Output**:
  - Real-time training log (last 50 lines visible)
  - Status updates (running/completed/error)
  - Success message: "✅ Eğitim başarıyla tamamlandı!"
  - Error message with traceback if training fails
  - Generated preview images from validation set

### Real-Time Output Display

- **Status Indicator**: Shows 🔄 (running) or ✅ (completed) or ❌ (error)
- **Log Window**: Displays last 50 lines of training output
- **Expandable**: View full training output in expandable section
- **Previews**: Shows generated segmentation/detection preview images

### Preview Images Section

- **Title**: "📸 Örnek Çıktılar (Validation)"
- **Content**: 2-column grid of validation output images
- **Source**: Generated from `outputs/previews_detection/` or `outputs/previews_segmentation/`

---

## Tab 5: Experiments

### Purpose

Monitor training progress, compare experiments, and manage previous training runs.

### Top Controls

#### **Task Type Filter Display**

- Shows current task type (🎯 Object Detection or 🔍 Segmentation)
- Changes based on sidebar selector

#### **🔄 Otomatik yenileme** (Auto Refresh) (Checkbox)

- **Purpose**: Automatically refresh metrics at set intervals
- **Default**: Unchecked

#### **Yenileme sıklığı** (Refresh Interval) (Slider - shown when auto-refresh enabled)

- **Purpose**: How often to refresh data
- **Range**: 5 to 60 seconds
- **Default**: 10 seconds

#### **🔄 Şimdi Yenile** (Refresh Now) (Button)

- **Function**: Manually refresh all experiment data immediately

---

### Experiments Table

Displays all training runs for the current task type with columns:

- **Seç** (Select): Checkbox to select multiple runs
- **Experiment Name**: Custom name given during training
- **run_id**: Unique identifier (shortened: first 8 chars + "...")
- **Start Time**: When training started (YYYY-MM-DD HH:MM:SS)
- **Task Type**: detection or segmentation
- **Epochs**: Number of training epochs configured
- **Learning Rate**: LR used during training
- **Model Name**: Model architecture used (truncated if long)
- **Train Loss**: Last recorded training loss
- **Val Loss**: Last recorded validation loss

**Sorting**: Automatically sorted by start time (newest first)

---

### Run Management

#### **🗑️ Seçilenleri Sil** (Delete Selected) (Secondary Button)

- **Function**: Delete selected training runs
- **Requirements**: At least one run must be selected
- **Confirmation**: Shows warning with count of runs to delete

#### **Delete Confirmation Dialog**

- **✅ Evet, Sil** (Yes, Delete): Confirms deletion
  - Deletes all selected runs from MLflow
  - Shows success count
  - Auto-refreshes table
- **❌ İptal** (Cancel): Cancels deletion and hides dialog

---

### Live Metrics Section

**Shown When**: Exactly one run is selected

#### **Run Info Header**

- Displays selected run name/ID

#### **Key Metrics Cards** (4 columns)

- **Train Loss**: Current loss with best value and epoch number
- **Val Loss**: Current loss with best value and epoch number
- **Epoch**: Current epoch / total epochs (e.g., "45/100")
- **Status**: 🟢 FINISHED | 🟡 RUNNING | 🔴 FAILED

#### **Loss Graphs**

##### **Canlı Loss Grafiği** (Live Loss Chart)

- **Axes**: Epoch (x) vs Loss (y)
- **Lines**:
  - Blue line with circles: Training loss
  - Orange dashed line with squares: Validation loss
- **Features**: Grid, legend, title with run name

##### **Tüm Metrikler** (All Metrics)

- Larger graph showing comprehensive metrics
- **Detection Task** (additional metrics):
  - Precision@0.5 (green line)
  - Recall@0.5 (red line)
  - F1@0.5 (purple line)
- **Segmentation Task** (additional metrics):
  - mIoU (mean Intersection over Union)
  - Pixel Accuracy
  - Other task-specific metrics
- **Features**: Multiple lines, different colors and styles, legend

---

### Multiple Runs Comparison

**Shown When**: 2 or more runs selected

- **Combined Comparison Table**: Shows metrics for all selected runs
- **Side-by-side Analysis**: Compare architectures, hyperparameters, and results
- **Interactive Features**: Can switch between individual and comparison views

---

## Data Flow & Recommended Workflow

### Typical Complete Workflow

```
1. Dataset Import Tab
   ↓ Download dataset from Hugging Face

2. Dataset Validation Tab
   ↓ Convert format (if needed)
   ↓ Reorganize structure (if needed)
   ↓ Rotate images (if needed)
   ↓ Validate dataset quality

3. Dataset Split Tab
   ↓ Set train/val/test ratios
   ↓ Create split files

4. Train Tab
   ↓ Select model and hyperparameters
   ↓ Start training
   ↓ Wait for completion
   ↓ Review preview outputs

5. Experiments Tab
   ↓ Monitor training progress
   ↓ Compare multiple runs
   ↓ Analyze metrics
   ↓ Delete unsuccessful runs
```

---

## Key Features Summary

| Feature                   | Location | Purpose                             |
| ------------------------- | -------- | ----------------------------------- |
| **Import from Hub**       | Tab 1    | Download datasets easily            |
| **Format Conversion**     | Tab 2    | Support multiple annotation formats |
| **Data Reorganization**   | Tab 2    | Simplify dataset structure          |
| **Image Rotation**        | Tab 2    | Fix orientation issues              |
| **Quality Validation**    | Tab 2    | Detect dataset problems             |
| **Train/Val/Test Split**  | Tab 3    | Create balanced subsets             |
| **Model Selection**       | Tab 4    | Choose from many architectures      |
| **Hyperparameter Tuning** | Tab 4    | Fine-tune training parameters       |
| **Data Augmentation**     | Tab 4    | Enhance training data               |
| **Real-time Monitoring**  | Tab 5    | Watch training progress             |
| **Metrics Visualization** | Tab 5    | Compare model performance           |
| **Run Management**        | Tab 5    | Organize and delete experiments     |

---

## Tips & Best Practices

1. **Always Backup**: Enable backup when converting dataset formats
2. **Start Small**: Use 10 epochs for testing before full training
3. **Reproducibility**: Keep the same random seed for consistent splits
4. **Memory**: Start with smaller batch sizes (4-8) if GPU runs out of memory
5. **Learning Rate**: Reduce if training loss increases; increase if it plateaus
6. **Augmentation**: Enable for small datasets, disable for large datasets
7. **Naming**: Use descriptive experiment names for easy tracking
8. **Monitoring**: Check Experiments tab regularly during training
9. **Cleanup**: Delete unsuccessful runs to keep MLflow organized
10. **Documentation**: Save experiment configs for reproducibility

---

## Troubleshooting

### Common Issues

| Problem              | Solution                                     |
| -------------------- | -------------------------------------------- |
| Dataset not found    | Use full path instead of relative path       |
| Ratio warning        | Adjust sliders so they sum to 1.0            |
| Training won't start | Ensure splits directory path is correct      |
| Out of memory        | Reduce batch size or model size              |
| No metrics displayed | Wait for training to progress, then refresh  |
| Missing annotations  | Run validation to identify problematic files |

---

## File Locations

- **Config**: `configs/train_config.yaml`
- **Splits**: `splits/` (train.txt, val.txt, test.txt)
- **MLflow Runs**: `mlruns/` directory
- **Outputs**: `outputs/previews_detection/` or `outputs/previews_segmentation/`
- **Datasets**: `DataSet/` (recommended location)

---

## Contact & Support

For issues or questions:

1. Check MLflow logs in `mlruns/` directory
2. Review full training output in Experiments tab
3. Validate dataset structure in Dataset Validation tab
4. Check console output for error messages
