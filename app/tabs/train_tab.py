import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

def run_hyperparameter_tuning(
    task_type: str,
    model_name: str,
    epochs_per_trial: int,
    fixed_batch_size: int = None,
    fixed_lr: float = None,
    fixed_weight_decay: float = None,
    preview_samples: int = 3,
    run_name: str = None,
    dataset_root: str = None,
    splits_dir: str = None,
    detection_labels_root: str = None,
    segmentation_masks_root: str = None,
    n_trials: int = 20,
    timeout_minutes: int = 120,
    optimize_metric: str = "f1",
    tune_lr: bool = True,
    tune_batch_size: bool = True,
    tune_weight_decay: bool = True,
    # Augmentation parameters
    augment: bool = False,
    augment_val: bool = False,
    horizontal_flip: float = 0.5,
    rotation: float = 10.0,
    scale: float = 0.1,
    translation: float = 0.1,
    crop_scale: float = 0.8,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    blur: float = 0.1,
    noise: float = 0.1,
    mixup: float = 0.0,
    cutmix: float = 0.0,
    mosaic: float = 0.0,
    elastic: float = 0.1,
    grid_distortion: float = 0.1,
):
    """Run hyperparameter tuning using Optuna."""
    
    # Check if Optuna is available
    try:
        import optuna
    except ImportError:
        st.error("❌ Optuna not found. Please install: pip install optuna")
        return
    
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = (project_root / "configs" / "train_config.yaml").resolve()
    
    if not config_path.exists():
        st.error(f"❌ Config file not found: {config_path}")
        return
    
    # Check if tuning module exists
    tune_module_path = project_root / "data" / "tune.py"
    if not tune_module_path.exists():
        st.error(f"❌ Tuning module not found: {tune_module_path}")
        return
    
    st.info("🔍 Starting hyperparameter tuning...")
    st.info(f"📊 Will run {n_trials} trials, up to {timeout_minutes} minutes")
    
    # Create placeholders for real-time output
    output_placeholder = st.empty()
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    try:
        # Set PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        # Run tuning script
        tuning_script = project_root / "data" / "tune.py"
        
        # Build command
        cmd = [
            sys.executable,
            str(tuning_script),
            "--task_type", task_type,
            "--model_name", model_name,
            "--epochs_per_trial", str(epochs_per_trial),
            "--n_trials", str(n_trials),
            "--timeout", str(timeout_minutes * 60),  # seconds
            "--optimize_metric", optimize_metric,
            "--splits_dir", splits_dir,
            "--preview_samples", str(preview_samples),
        ]
        
        if run_name:
            cmd.extend(["--run_name", run_name])
        if fixed_batch_size:
            cmd.extend(["--fixed_batch_size", str(fixed_batch_size)])
        if fixed_lr:
            cmd.extend(["--fixed_lr", str(fixed_lr)])
        if fixed_weight_decay:
            cmd.extend(["--fixed_weight_decay", str(fixed_weight_decay)])
        
        cmd.extend(["--tune_lr", str(tune_lr)])
        cmd.extend(["--tune_batch_size", str(tune_batch_size)])
        cmd.extend(["--tune_weight_decay", str(tune_weight_decay)])
        
        # Optional dataset/labels override
        if dataset_root:
            cmd.extend(["--dataset_root", dataset_root])
        if detection_labels_root:
            cmd.extend(["--detection_labels_root", detection_labels_root])
        if segmentation_masks_root:
            cmd.extend(["--segmentation_masks_root", segmentation_masks_root])
        
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        
        output_lines = []
        with status_placeholder.container():
            st.info("🔄 Hyperparameter tuning in progress... (Live logs below)")
        
        for line in process.stdout:
            output_lines.append(line.rstrip())
            with output_placeholder.container():
                st.code('\n'.join(output_lines[-50:]))
        
        process.wait()
        
        if process.returncode == 0:
            status_placeholder.success("✅ Hyperparameter tuning completed successfully!")
            with st.expander("Show all tuning output", expanded=False):
                st.code('\n'.join(output_lines))
            
            # Show best parameters
            st.markdown("### 🏆 Best Hyperparameters")
            st.info("💡 Best parameters saved in MLflow under the 'tuning_best' run.")
        else:
            status_placeholder.error("❌ Hyperparameter tuning ended with an error!")
            with st.expander("Show error output", expanded=True):
                st.code('\n'.join(output_lines))
                
    except Exception as e:
        status_placeholder.error(f"❌ Tuning execution error: {e}")
        import traceback
        st.code(traceback.format_exc())




def run_training(
    task_type: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    preview_samples: int = 3,
    run_name: str = None,
    dataset_root: str = None,
    splits_dir: str = None,
    detection_labels_root: str = None,
    segmentation_masks_root: str = None,
    weight_decay: float = 0.0005,
    # Augmentation parameters
    augment: bool = False,
    augment_val: bool = False,
    horizontal_flip: float = 0.5,
    rotation: float = 10.0,
    scale: float = 0.1,
    translation: float = 0.1,
    crop_scale: float = 0.8,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    blur: float = 0.1,
    noise: float = 0.1,
    mixup: float = 0.0,
    cutmix: float = 0.0,
    mosaic: float = 0.0,
    elastic: float = 0.1,
    grid_distortion: float = 0.1,
):
    """Write overrides to configs/train_config.yaml and kick off training as a subprocess."""

    # Get project root and use absolute path for config
    # __file__ = app/tabs/train_tab.py -> parent.parent.parent = repo root
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = (project_root / "configs" / "train_config.yaml").resolve()

    # Check if config file exists
    if not config_path.exists():
        error_msg = (
            f"❌ Config file not found: {config_path}\n\n"
            f"Checked locations:\n"
            f"  - {config_path}\n"
            f"  - {project_root / 'configs' / 'train_config.yaml'}\n"
            f"  - {Path('configs') / 'train_config.yaml'}\n\n"
            f"Project root: {project_root}\n"
            f"Current working directory: {os.getcwd()}\n\n"
            f"Please:\n"
            f"  1. Run Streamlit from the project root: streamlit run app/main.py\n"
            f"  2. Ensure configs/train_config.yaml exists"
        )
        raise FileNotFoundError(error_msg)

    # Simple in-place override for quick experimentation
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["task_type"] = task_type
    cfg["common"]["epochs"] = epochs
    cfg["common"]["batch_size"] = batch_size
    cfg["common"]["lr"] = lr
    cfg["common"]["weight_decay"] = weight_decay
    cfg["common"]["preview_samples"] = int(preview_samples)

    # Update splits directory (required)
    if splits_dir:
        cfg["data"]["splits_dir"] = splits_dir

    # Update optional paths if provided
    if detection_labels_root:
        cfg["data"]["detection_labels_root"] = detection_labels_root
    if segmentation_masks_root:
        cfg["data"]["segmentation_masks_root"] = segmentation_masks_root
    # dataset_root is now inferred automatically, but can be set if needed
    if dataset_root:
        cfg["data"]["dataset_root"] = dataset_root

    if task_type == "detection":
        cfg.setdefault("detection", {})
        cfg["detection"]["model_name"] = model_name
    else:
        cfg.setdefault("segmentation", {})
        cfg["segmentation"]["model_name"] = model_name

    # Add run_name to config if provided
    if run_name:
        cfg["run_name"] = run_name

    # Add augmentation config
    cfg.setdefault("augmentation", {})
    aug_cfg = cfg["augmentation"]
    aug_cfg["augment"] = augment
    aug_cfg["augment_val"] = augment_val
    aug_cfg["horizontal_flip"] = horizontal_flip
    aug_cfg["rotation"] = rotation
    aug_cfg["scale"] = scale
    aug_cfg["translation"] = translation
    aug_cfg["crop_scale"] = crop_scale
    aug_cfg["brightness"] = brightness
    aug_cfg["contrast"] = contrast
    aug_cfg["saturation"] = saturation
    aug_cfg["hue"] = hue
    aug_cfg["blur"] = blur
    aug_cfg["noise"] = noise
    aug_cfg["mixup"] = mixup
    aug_cfg["cutmix"] = cutmix
    aug_cfg["mosaic"] = mosaic
    aug_cfg["elastic"] = elastic
    aug_cfg["grid_distortion"] = grid_distortion

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Run train.py from project root directory with PYTHONPATH set
    st.info("⏳ Starting training... (Please wait)")
    
    # Set PYTHONPATH to project root so imports work
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # Create placeholders for real-time output
    output_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            [sys.executable, str(project_root / "data" / "train.py"), "--config", str(config_path)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of crashing
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        
        # Collect output and display in real-time
        output_lines = []
        with status_placeholder.container():
            st.info("🔄 Training in progress... (Live logs below)")
        
        for line in process.stdout:
            output_lines.append(line.rstrip())
            # Show last 50 lines in real-time
            with output_placeholder.container():
                st.code('\n'.join(output_lines[-50:]))
        
        process.wait()
        
        if process.returncode == 0:
            status_placeholder.success("✅ Training completed successfully!")
            with st.expander("Show all training output", expanded=False):
                st.code('\n'.join(output_lines))

            # Show previews if generated
            run_name_cfg = cfg.get("run_name") or "default_run"
            safe_run_name = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run_name_cfg))
            preview_dir = project_root / "outputs" / (
                "previews_detection" if task_type == "detection" else "previews_segmentation"
            ) / safe_run_name
            # Fallback to old flat layout if needed
            preview_dirs_to_check = [preview_dir, preview_dir.parent]
            preview_files = []
            for pdir in preview_dirs_to_check:
                if pdir.exists():
                    preview_files = sorted(pdir.glob("*.png"))
                    if preview_files:
                        break
            if preview_files:
                if preview_files:
                    st.markdown("### 📸 Sample Outputs (Validation)")
                    show_n = min(len(preview_files), int(cfg["common"].get("preview_samples", 0) or len(preview_files)))
                    cols = st.columns(2)
                    for idx, img_path in enumerate(preview_files[:show_n]):
                        with cols[idx % 2]:
                            st.image(str(img_path), caption=img_path.name, use_column_width=True)
                else:
                    st.info("No preview images found. Ensure preview_samples > 0.")
        else:
            status_placeholder.error("❌ Training ended with an error!")
            with st.expander("Show error output", expanded=True):
                st.code('\n'.join(output_lines))
                
    except Exception as e:
        status_placeholder.error(f"❌ Training execution error: {e}")


def render_train_tab(task_type: str, task_display: str) -> None:
    """Render the Train tab UI and kick off training when requested."""

    st.subheader(f"Start new training ({task_display})")

    # Dataset configuration - only the splits folder
    st.markdown("### Dataset Settings")
    col_ds1, col_ds2 = st.columns(2)
    with col_ds1:
        splits_dir = st.text_input(
            "Splits directory (train.txt, val.txt, test.txt)", value="splits", key="train_splits_dir"
        )
    with col_ds2:
        dataset_root = st.text_input(
            "Dataset root (folder with images_all/labels_all)", value="", key="train_dataset_root",
            help="Example: DataSet/obj. Leave empty to infer automatically."
        )

    # Apply dataset settings to configs/train_config.yaml
    if st.button("Apply dataset settings to config", help="Writes splits_dir and dataset_root into configs/train_config.yaml"):
        try:
            import yaml
            project_root = Path(__file__).parent.parent.parent.resolve()
            config_path = (project_root / "configs" / "train_config.yaml").resolve()
            if not config_path.exists():
                st.error(f"❌ Config file not found: {config_path}")
            else:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("data", {})
                if splits_dir:
                    cfg["data"]["splits_dir"] = splits_dir
                if dataset_root:
                    cfg["data"]["dataset_root"] = dataset_root
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f)
                st.success("✅ Dataset settings saved to configs/train_config.yaml")
        except Exception as e:
            st.error(f"❌ Failed to update config: {e}")

    # Optional: labels/masks dizinleri (otomatik bulunamazsa)
    with st.expander("Advanced Settings (Optional)"):
        if task_type == "detection":
            detection_labels_root = st.text_input(
                "Detection labels directory (auto-discovered if left blank)",
                value="",
                key="train_detection_labels",
            )
            if not detection_labels_root:
                detection_labels_root = None
        else:
            detection_labels_root = None

        if task_type == "segmentation":
            segmentation_masks_root = st.text_input(
                    "Segmentation masks directory (auto-discovered if left blank)",
                value="",
                key="train_segmentation_masks",
            )
            if not segmentation_masks_root:
                segmentation_masks_root = None
        else:
            segmentation_masks_root = None

    st.markdown("### Model and Hyperparameter Settings")

    # Experiment name input
    experiment_name = st.text_input(
        "Experiment name (optional)",
        value="",
        placeholder="e.g., deeplabv3_batch4_lr0.0005",
        key="experiment_name",
    )
    
    # Hyperparameter tuning seçeneği
    tuning_mode = st.radio(
        "Hyperparameter Mode",
        ["Manual", "Automatic Tuning"],
        help="Manual: Set hyperparameters yourself. Automatic Tuning: Use Optuna to find the best hyperparameters.",
        key="tuning_mode"
    )
    
    if task_type == "detection":
        model_name = st.selectbox(
            "Detection model",
            [
                "yolov8n",
                "yolov8s",
                "yolov8m",
                "yolov8l",
                "yolov8x",
                "yolov5n",
                "yolov5s",
                "yolov5m",
                "yolov5l",
                "yolov5x",
                "detr",
                "cascade_rcnn",
                "efficientdet_d0",
                "efficientdet_d1",
                "efficientdet_d2",
                "fcos",
                "atss",
                "fasterrcnn_resnet50_fpn",
                "fasterrcnn_mobilenet_v3_large_fpn",
                "retinanet_resnet50_fpn",
                "ssd300_vgg16",
            ],
        )
    else:
        model_name = st.selectbox(
            "Segmentation model",
            [
                "mask2former",
                "segnext",
                "bisenetv2",
                "ddrnet",
                "pidnet",
                "topformer",
                "segformer",
                "deeplabv3+",
                "unet",
                "unet++",
                "pspnet",
                "hrnet",
                "deeplabv3_resnet50",
                "fcn_resnet50",
            ],
        )
    
    # Show UI based on Manual or Automatic Tuning mode
    if tuning_mode == "Manual":
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10)
        batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=4)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=5e-4, format="%.6f")
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=0.01, value=0.0005, format="%.6f", step=0.0001)
    else:
        # Automatic Tuning mode
        st.info("🔍 Automatic Tuning: Optuna will search for the best hyperparameters.")
        
        col_tune1, col_tune2 = st.columns(2)
        with col_tune1:
            n_trials = st.number_input(
                "Number of trials (n_trials)",
                min_value=5,
                max_value=100,
                value=20,
                help="How many different hyperparameter combinations will be tried"
            )
            tuning_timeout = st.number_input(
                "Maximum duration (minutes)",
                min_value=10,
                max_value=1440,
                value=120,
                help="Maximum duration for tuning (minutes)"
            )
        
        with col_tune2:
            optimize_metric = st.selectbox(
                "Metric to optimize",
                ["f1", "miou", "val_loss"] if task_type == "detection" else ["miou", "pixel_acc", "val_loss"],
                help="Which metric will be used to select best hyperparameters"
            )
            epochs_per_trial = st.number_input(
                "Epochs per trial",
                min_value=3,
                max_value=50,
                value=5,
                help="Number of epochs to train for each hyperparameter combination"
            )
        
        # Parameters to tune
        st.markdown("#### Parameters to Tune")
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            tune_lr = st.checkbox("Learning rate", value=True)
            tune_batch_size = st.checkbox("Batch size", value=True)
            tune_weight_decay = st.checkbox("Weight decay", value=True)
        
        with col_param2:
            # Fixed values (for parameters not tuned)
            st.caption("Fixed values (not tuned):")
            batch_size = st.number_input("Batch size (fixed)", min_value=1, max_value=64, value=4, key="tune_batch_fixed", disabled=tune_batch_size)
            lr = st.number_input("Learning rate (fixed)", min_value=1e-6, max_value=1e-1, value=5e-4, format="%.6f", key="tune_lr_fixed", disabled=tune_lr)
            weight_decay = st.number_input("Weight decay (fixed)", min_value=0.0, max_value=0.01, value=0.0005, format="%.6f", key="tune_wd_fixed", disabled=tune_weight_decay)
    
    preview_samples = st.slider(
        "Number of validation images to show after training",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="After training, log ground truth/prediction images.",
    )

    # Augmentation settings
    st.markdown("### Augmentation Settings")

    col_aug1, col_aug2 = st.columns(2)
    with col_aug1:
        augment = st.checkbox(
            "Enable augmentation",
            value=False,
            help="Use augmentation techniques to increase training data",
        )
        augment_val = st.checkbox(
            "Use for validation as well",
            value=False,
            help="Apply augmentation to validation set (typically off)",
        )

    with col_aug2:
        st.write("**Dataset Enlargement Ratio:**")
        if augment:
            st.info("🎨 Dataset can be enlarged 2-10x")
        else:
            st.info("❌ Dataset enlargement disabled")

    if augment:
        with st.expander("🎨 Augmentation Details"):
            # Initialize all variables with defaults first
            horizontal_flip = rotation = scale = translation = crop_scale = 0.0
            brightness = contrast = saturation = hue = blur = noise = 0.0
            mixup = cutmix = mosaic = elastic = grid_distortion = 0.0
            
            st.markdown("#### Geometric Transformations")
            col_geo1, col_geo2, col_geo3 = st.columns(3)
            with col_geo1:
                horizontal_flip = st.slider(
                    "Horizontal flip", 0.0, 1.0, 0.5, 0.1, help="Probability of horizontally flipping images"
                )
                rotation = st.slider(
                    "Rotation", 0.0, 45.0, 10.0, 1.0, help="Maximum rotation angle"
                )
            with col_geo2:
                scale = st.slider("Scaling", 0.0, 0.5, 0.1, 0.05, help="Scaling range")
                translation = st.slider("Translation", 0.0, 0.5, 0.1, 0.05, help="Translation range")
            with col_geo3:
                crop_scale = st.slider(
                    "Crop scale", 0.5, 1.0, 0.8, 0.05, help="Minimum scale for random cropping"
                )

            st.markdown("#### Color Transformations")
            col_color1, col_color2, col_color3 = st.columns(3)
            with col_color1:
                brightness = st.slider("Brightness", 0.0, 1.0, 0.2, 0.1, help="Brightness change range")
                contrast = st.slider("Contrast", 0.0, 1.0, 0.2, 0.1, help="Contrast change range")
            with col_color2:
                saturation = st.slider("Saturation", 0.0, 1.0, 0.2, 0.1, help="Saturation change range")
                hue = st.slider("Hue", 0.0, 0.5, 0.1, 0.05, help="Hue change range")
            with col_color3:
                blur = st.slider("Blur", 0.0, 1.0, 0.1, 0.1, help="Probability of blurring")
                noise = st.slider("Noise", 0.0, 1.0, 0.1, 0.1, help="Probability of adding noise")

            if task_type == "detection":
                st.markdown("#### Advanced Detection")
                col_adv1, col_adv2, col_adv3 = st.columns(3)
                with col_adv1:
                    mixup = st.slider("Mixup", 0.0, 1.0, 0.0, 0.1, help="Probability of mixup augmentation")
                with col_adv2:
                    cutmix = st.slider("CutMix", 0.0, 1.0, 0.0, 0.1, help="Probability of CutMix augmentation")
                with col_adv3:
                    mosaic = st.slider("Mosaic", 0.0, 1.0, 0.0, 0.1, help="Probability of Mosaic augmentation")

            if task_type == "segmentation":
                st.markdown("#### Segmentation Specific")
                col_seg1, col_seg2 = st.columns(2)
                with col_seg1:
                    elastic = st.slider(
                        "Elastic deformation", 0.0, 1.0, 0.1, 0.1, help="Probability of elastic deformation"
                    )
                with col_seg2:
                    grid_distortion = st.slider(
                        "Grid distortion", 0.0, 1.0, 0.1, 0.1, help="Probability of grid distortion"
                    )
    else:
        # Default values
        horizontal_flip = rotation = scale = translation = crop_scale = 0.0
        brightness = contrast = saturation = hue = blur = noise = 0.0
        mixup = cutmix = mosaic = elastic = grid_distortion = 0.0

    if st.button("Train" if tuning_mode == "Manual" else "🔍 Start Tuning", type="primary"):
        # Validate splits directory
        splits_path = Path(splits_dir)
        if not splits_path.exists():
            st.error(f"❌ Splits directory not found: {splits_dir}")
            return

        train_file = splits_path / "train.txt"
        val_file = splits_path / "val.txt"
        if not train_file.exists():
            st.error(f"❌ train.txt not found: {train_file}")
            return
        if not val_file.exists():
            st.warning(f"⚠️ val.txt not found: {val_file}. Only train.txt will be used.")

        # Validate dataset_root if provided
        dataset_root_val = dataset_root.strip() or None
        if dataset_root_val:
            ds_path = Path(dataset_root_val)
            if not ds_path.exists():
                st.error(f"❌ Dataset root not found: {ds_path}")
                return
            # Optional sanity check for expected subfolders
            has_expected = (ds_path / "images_all").exists() or (ds_path / "images").exists()
            if not has_expected:
                st.warning(f"⚠️ '{ds_path}' does not contain images_all/ or images/. Training may fail.")

        st.info(f"📂 Splits directory: {splits_dir}")
        if dataset_root_val:
            st.info(f"📁 Dataset root: {dataset_root_val}")
        else:
            st.info("🔍 Dataset root will be inferred automatically from paths in train.txt.")
        
        if tuning_mode == "Manual":
            st.write("Training started in background...")
            run_training(
                task_type,
                model_name,
                int(epochs),
                int(batch_size),
                float(lr),
                int(preview_samples),
                run_name=experiment_name if experiment_name else None,
                dataset_root=dataset_root_val,
                splits_dir=splits_dir,
                detection_labels_root=detection_labels_root if task_type == "detection" else None,
                segmentation_masks_root=segmentation_masks_root if task_type == "segmentation" else None,
                weight_decay=float(weight_decay),
                # Augmentation parameters
                augment=augment,
                augment_val=augment_val,
                horizontal_flip=horizontal_flip,
                rotation=rotation,
                scale=scale,
                translation=translation,
                crop_scale=crop_scale,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                blur=blur,
                noise=noise,
                mixup=mixup,
                cutmix=cutmix,
                mosaic=mosaic,
                elastic=elastic,
                grid_distortion=grid_distortion,
            )
        else:
            # Automatic Tuning
            st.write("🔍 Starting hyperparameter tuning...")
            run_hyperparameter_tuning(
                task_type,
                model_name,
                int(epochs_per_trial),
                int(batch_size) if not tune_batch_size else None,
                float(lr) if not tune_lr else None,
                float(weight_decay) if not tune_weight_decay else None,
                int(preview_samples),
                run_name=experiment_name if experiment_name else None,
                dataset_root=dataset_root_val,
                splits_dir=splits_dir,
                detection_labels_root=detection_labels_root if task_type == "detection" else None,
                segmentation_masks_root=segmentation_masks_root if task_type == "segmentation" else None,
                n_trials=int(n_trials),
                timeout_minutes=int(tuning_timeout),
                optimize_metric=optimize_metric,
                tune_lr=tune_lr,
                tune_batch_size=tune_batch_size,
                tune_weight_decay=tune_weight_decay,
                # Augmentation parameters
                augment=augment,
                augment_val=augment_val,
                horizontal_flip=horizontal_flip,
                rotation=rotation,
                scale=scale,
                translation=translation,
                crop_scale=crop_scale,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                blur=blur,
                noise=noise,
                mixup=mixup,
                cutmix=cutmix,
                mosaic=mosaic,
                elastic=elastic,
                grid_distortion=grid_distortion,
            )
