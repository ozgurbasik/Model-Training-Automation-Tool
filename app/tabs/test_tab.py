"""
Test tab for evaluating trained models on test dataset.
"""
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import streamlit as st
import yaml


def render_test_tab(task_type: str) -> None:
    """Render test tab for model evaluation."""
    
    st.header("🧪 Model Testing")
    st.caption("Evaluate trained models on the test dataset")
    
    task_display = "🎯 Object Detection" if task_type == "detection" else "🔍 Segmentation"
    st.info(f"📋 Task type: **{task_display}**")

    # Load defaults from configs/test_config.yaml if present
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = (project_root / "configs" / "test_config.yaml").resolve()
    test_cfg = {}
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                test_cfg = yaml.safe_load(f) or {}
    except Exception as e:
        st.warning(f"⚠️ Could not read config defaults: {e}")
        test_cfg = {}
    
    # Model selection
    st.subheader("📦 Model Selection")
    
    output_dir = Path("outputs")
    if not output_dir.exists():
        st.error("❌ outputs/ folder not found. Train a model first.")
        return
    
    # Run klasörlerini bul
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        st.error("❌ No trained models found. Train a model first.")
        return
    
    run_names = [d.name for d in run_dirs]
    selected_run = st.selectbox("Select a run", run_names, key="test_run_select")
    
    if not selected_run:
        return
    
    run_dir = output_dir / selected_run
    if not run_dir.exists():
        # YOLO may auto-increment run dirs (e.g., fresh_test2). Pick latest matching prefix.
        prefix_matches = [d for d in run_dirs if d.name == selected_run or d.name.startswith(f"{selected_run}")]
        if prefix_matches:
            run_dir = max(prefix_matches, key=lambda p: p.stat().st_mtime)
            st.info(f"ℹ️ Run folder {selected_run} not found, using the most recent matching folder: {run_dir.name}")
        else:
            st.error(f"❌ Run folder not found: {selected_run}.")
            return
    
    # Model dosyalarını bul
    if task_type == "detection":
        model_files = {
            "Best Model": run_dir / "detection_best.pt",
            "Last Model": run_dir / "detection_last.pt",
            "YOLO Best": run_dir / "yolo_best.pt",
        }
        # Also look inside weights/ for ultralytics best/last
        weights_dir = run_dir / "weights"
        model_files["YOLO weights/best.pt"] = weights_dir / "best.pt"
        model_files["YOLO weights/last.pt"] = weights_dir / "last.pt"
    else:
        model_files = {
            "Best Model": run_dir / "segmentation_best.pt",
            "Last Model": run_dir / "segmentation_last.pt",
        }
    
    # Var olan modelleri filtrele
    available_models = {k: v for k, v in model_files.items() if v.exists()}
    
    if not available_models:
        st.error(f"❌ No model files found in run: {selected_run}.")
        return
    
    selected_model_type = st.selectbox("Select model type", list(available_models.keys()), key="test_model_type")
    model_path = available_models[selected_model_type]
    
    # Test dataset path
    st.subheader("📂 Test Dataset")

    default_splits_dir = Path(test_cfg.get("splits_dir", "splits"))
    default_dataset_root = test_cfg.get("dataset_root") or ""
    default_test_labels_dir_cfg = test_cfg.get("test_labels_dir") or ""

    dataset_root_input = st.text_input(
        "Dataset root (optional)",
        value=str(Path(default_dataset_root).resolve()) if default_dataset_root else "",
        help="Root folder containing images_all/labels_all. Leave blank to skip.",
        key="test_dataset_root_input",
    )
    
    # Splits klasörü seçimi
    splits_dir_input = st.text_input(
        "Splits folder path",
        value=str(default_splits_dir.resolve()),
        help="Full path to the splits folder. The system will automatically find test/images and test/labels.",
        key="splits_dir_input"
    )

    if st.button(
        "Apply dataset settings to config",
        help="Writes splits_dir, dataset_root, and test_labels_dir into configs/test_config.yaml",
        key="test_apply_dataset_settings",
    ):
        try:
            cfg_to_update = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_to_update = yaml.safe_load(f) or {}
            if splits_dir_input:
                cfg_to_update["splits_dir"] = splits_dir_input
            if dataset_root_input:
                cfg_to_update["dataset_root"] = dataset_root_input
            # Save test_labels_dir for detection
            if task_type == "detection" and "test_labels_dir_input" in locals():
                cfg_to_update["test_labels_dir"] = test_labels_dir_input
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_to_update, f)
            st.success("✅ Dataset settings saved to configs/test_config.yaml")
        except Exception as e:
            st.error(f"❌ Failed to update config: {e}")
    
    splits_dir = Path(splits_dir_input)
    
    # Splits klasörü kontrolü
    if not splits_dir.exists():
        st.error(f"❌ Splits folder not found: {splits_dir}")
        st.info("💡 Please enter a valid splits folder path. Example: D:\\RcCarModelTraining\\splits")
        return
    
    # Test klasörünü otomatik bul
    test_dir = splits_dir / "test"
    test_images_dir = test_dir / "images"
    # Detection labels may live elsewhere (YOLO txt). Allow override for detection; segmentation keeps default.
    default_test_labels_dir = Path(default_test_labels_dir_cfg) if default_test_labels_dir_cfg else (test_dir / "labels")
    
    # Allow custom test images directory
    test_images_dir_input = st.text_input(
        "Test images folder",
        value=str(test_images_dir.resolve()),
        help="Full path to the folder containing test images.",
        key="test_images_dir_input",
    )
    test_images_dir = Path(test_images_dir_input)
    
    # Test labels folder (for both detection and segmentation)
    task_label = "detection" if task_type == "detection" else "segmentation"
    test_labels_dir_input = st.text_input(
        "Test labels folder",
        value=str(default_test_labels_dir.resolve()),
        help="Folder containing test labels (YOLO txt for detection, masks for segmentation).",
        key="test_labels_dir_input",
    )
    test_labels_dir = Path(test_labels_dir_input)
    
    # Test images kontrolü
    if not test_images_dir.exists():
        st.error(f"❌ Test images folder not found: {test_images_dir}")
        st.info("💡 Please enter a valid test images folder path.")
        return
    
    # Read test.txt to get list of test images
    test_images = []
    test_list_file = splits_dir / "test.txt"
    
    if test_list_file.exists():
        try:
            with open(test_list_file, "r", encoding="utf-8") as f:
                test_list_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            
            # Convert relative paths to absolute paths
            for line in test_list_lines:
                img_path = splits_dir / line
                if img_path.exists():
                    test_images.append(img_path)
                else:
                    alt_path = test_images_dir / Path(line).name
                    if alt_path.exists():
                        test_images.append(alt_path)
            
            if test_images:
                st.info(f"📄 Using test.txt: {len(test_images)} test images found")
                st.caption(f"📁 {test_list_file}")
        except Exception as e:
            st.warning(f"⚠️ Could not read test.txt: {e}")
            test_images = []
    
    # Fallback: if test.txt does not exist or is empty, scan test_images_dir
    if not test_images:
        test_images = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.jpeg"))
        if test_images:
            st.warning(f"⚠️ test.txt not found. Scanning folder instead: {len(test_images)} images found")
            st.caption(f"📁 {test_images_dir}")
    
    if not test_images:
        st.error(f"❌ No images found in folder: {test_images_dir}.")
        return
    
    # Test labels kontrolü (detection için seçilen dizin zorunlu)
    if task_type == "detection" and not test_labels_dir.exists():
        st.error(f"❌ Test labels folder not found: {test_labels_dir}")
        st.info("💡 Make sure you entered the correct YOLO txt labels folder path.")
        return

    test_labels = []
    if test_labels_dir.exists():
        test_labels = list(test_labels_dir.glob("*.json")) + list(test_labels_dir.glob("*.txt"))
    
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        st.success(f"✅ Found {len(test_images)} test images")
        st.caption(f"📁 {test_images_dir}")
    with col_test2:
        if test_labels:
            st.info(f"📋 Found {len(test_labels)} test labels")
            st.caption(f"📁 {test_labels_dir}")
        else:
            st.warning("⚠️ Test labels folder not found or empty")
            st.caption("💡 Without labels, only inference will be performed")

    # Detection için test listesi (test.txt) kullan
    predefined_test_list: Optional[List[str]] = None
    if task_type == "detection":
        test_list_file = splits_dir / "test.txt"
        if test_list_file.exists():
            try:
                lines = [ln.strip() for ln in test_list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                predefined_test_list = lines if lines else None
                if predefined_test_list:
                    st.info(f"📄 Found test.txt and will use it ({len(predefined_test_list)} lines)")
                    st.caption(f"📁 {test_list_file}")
            except Exception as e:
                st.warning(f"⚠️ Could not read test.txt: {e}")
    
    # Model parametreleri (MLflow'dan veya config'den)
    st.subheader("⚙️ Model Parameters")
    
    # Try to infer model name from run folder name
    inferred_model_name = "unet" if task_type == "segmentation" else "yolov8n"
    if selected_run:
        # Parse run folder name to extract model name
        # Format: modelname_Epochs__X__LR__Y__Weight_Decay__Z
        parts = selected_run.split("_")
        if parts and not parts[0].lower().startswith(("epochs", "lr", "weight")):
            potential_model = parts[0].lower()
            # Common segmentation models
            if task_type == "segmentation" and any(m in potential_model for m in ["unet", "deeplab", "fcn", "pspnet"]):
                inferred_model_name = potential_model
            # Common detection models
            elif task_type == "detection" and any(m in potential_model for m in ["yolo", "faster", "rcnn", "ssd"]):
                inferred_model_name = potential_model
    
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.text_input(
            "Model name",
            value=inferred_model_name,
            help="Must match the architecture used during training. Check the run folder name for hints.",
            key="test_model_name"
        )
        if inferred_model_name != ("unet" if task_type == "segmentation" else "yolov8n"):
            st.info(f"💡 Inferred from run folder: {inferred_model_name}")
        img_size = st.number_input("Image size", min_value=128, max_value=2048, value=640, step=32, key="test_img_size")
    
    with col2:
        batch_size = st.number_input("Batch size", min_value=1, max_value=32, value=4, key="test_batch_size")
        
        # Sınıf sayısını classes.txt'den bul veya manuel gir
        default_classes_file = splits_dir / "train" / "labels" / "classes.txt"
        classes_path_input = st.text_input(
            "classes.txt path",
            value=str(default_classes_file.resolve()),
            help="Full path to classes.txt. Example: C:/Users/basik/Desktop/OzgurLocal/RC_CV/RC-Car-Model-Training/DataSet/RcCArDataset/labels/classes.txt",
            key="classes_file_path",
        )
        classes_file = Path(classes_path_input)
        num_classes = None
        class_names = []

        # Manual entry area (one class name per line)
        manual_classes_text = st.text_area(
            "Enter class names (one per line)",
            value="",
            placeholder="car\npedestrian\nbus",
            help="If classes.txt is missing, enter manually here; if present, it will be read automatically.",
            key="manual_class_names",
        )
        manual_class_list = [line.strip() for line in manual_classes_text.splitlines() if line.strip()]

        if classes_file.exists():
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    num_classes = len(lines)
                    for line in lines:
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            class_names.append(parts[1])
                        else:
                            class_names.append(parts[0])
                if num_classes > 0:
                    st.success(f"📋 Number of classes auto-detected: {num_classes}")
                    st.caption(f"📁 {classes_file}")
                    with st.expander("📋 Class List"):
                        for i, name in enumerate(class_names):
                            st.text(f"  {i}: {name}")
            except Exception as e:
                st.warning(f"⚠️ Could not read classes.txt: {e}")

        # Manual override if provided
        if manual_class_list:
            class_names = manual_class_list
            num_classes = len(class_names)
            st.info(f"✏️ Using manual class list ({num_classes} classes)")

        if num_classes is None or num_classes == 0:
            num_classes = st.number_input(
                "Number of classes (auto-detection failed)",
                min_value=1,
                max_value=100,
                value=4 if task_type == "segmentation" else 1,
                key="test_num_classes"
            )
            if classes_file.exists():
                st.info(f"💡 For class count, see: {classes_file}")
            else:
                st.warning(f"⚠️ {classes_file} not found. You can enter manually.")
    
    # Check and generate masks for segmentation if needed
    if task_type == "segmentation":
        test_masks_dir = splits_dir / "test" / "labels"
        test_json_dir = splits_dir / "test" / "labels"
        
        # Check if masks exist
        masks_exist = test_masks_dir.exists() and list(test_masks_dir.glob("*.png"))
        
        if not masks_exist and test_json_dir.exists() and list(test_json_dir.glob("*.json")):
            st.warning("⚠️ Test masks not found. Generating from JSON annotations...")
            
            # Generate masks
            try:
                import json
                import numpy as np
                from PIL import Image, ImageDraw
                
                # Load classes
                label_map = {}
                for cls_name in class_names:
                    label_map[cls_name] = class_names.index(cls_name)
                
                if not label_map:
                    st.error("❌ Cannot generate masks: no class names available")
                else:
                    test_masks_dir.mkdir(parents=True, exist_ok=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    json_files = list(test_json_dir.glob("*.json"))
                    for idx, json_path in enumerate(json_files):
                        with open(json_path, 'r') as f:
                            anno = json.load(f)
                        
                        # Get image dimensions
                        img_path = splits_dir / "test" / "images" / f"{json_path.stem}.png"
                        if not img_path.exists():
                            img_path = splits_dir / "test" / "images" / f"{json_path.stem}.jpg"
                        
                        if img_path.exists():
                            img = Image.open(img_path)
                            w, h = img.size
                        else:
                            h = anno.get("imageHeight", 640)
                            w = anno.get("imageWidth", 640)
                        
                        # Create blank mask
                        mask = np.zeros((h, w), dtype=np.uint8)
                        
                        # Draw shapes
                        shapes = anno.get("shapes", [])
                        for shape in shapes:
                            label = shape.get("label", "")
                            if not label or label not in label_map:
                                continue
                            
                            class_id = label_map[label]
                            points = shape.get("points", [])
                            
                            if len(points) < 3:
                                continue
                            
                            polygon_points = [(int(x), int(y)) for x, y in points]
                            mask_img = Image.fromarray(mask)
                            draw = ImageDraw.Draw(mask_img)
                            draw.polygon(polygon_points, fill=class_id)
                            mask = np.array(mask_img)
                        
                        # Save mask
                        out_path = test_masks_dir / f"{json_path.stem}.png"
                        Image.fromarray(mask).save(out_path)
                        
                        progress_bar.progress((idx + 1) / len(json_files))
                        status_text.text(f"Generated {idx + 1}/{len(json_files)} masks")
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"✅ Generated {len(json_files)} test masks in {test_masks_dir}")
                    
            except Exception as e:
                st.error(f"❌ Failed to generate masks: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Test button
    if st.button("🚀 Start Test", type="primary", use_container_width=True):
        # Get project root
        project_root = Path(__file__).parent.parent.parent.resolve()
        
        # Build command arguments
        cmd = [
            sys.executable,
            str(project_root / "data" / "test.py"),
            "--task_type", task_type,
            "--model_path", str(model_path),
            "--model_name", model_name,
            "--splits_dir", str(splits_dir),
            "--num_classes", str(num_classes),
            "--img_size", str(img_size),
            "--batch_size", str(batch_size),
            "--output_dir", str(run_dir),
        ]
        
        # Add test labels dir for detection
        if task_type == "detection":
            cmd.extend(["--test_labels_dir", str(test_labels_dir)])
        
        st.info(f"📁 Test preview images will be saved to: {run_dir / 'test_preview'}")
        st.info("⏳ Starting test... (Please wait)")
        
        # Set PYTHONPATH to project root
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        # Create placeholders for real-time output
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # Use Popen for real-time output streaming
            import subprocess
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
            
            # Collect output and display in real-time
            output_lines = []
            with status_placeholder.container():
                st.info("🔄 Test in progress... (Live logs below)")
            
            for line in process.stdout:
                output_lines.append(line.rstrip())
                # Show last 50 lines in real-time
                with output_placeholder.container():
                    st.code('\n'.join(output_lines[-50:]))
            
            process.wait()
            
            if process.returncode == 0:
                status_placeholder.success("✅ Test completed successfully!")
                with st.expander("Show all test output", expanded=False):
                    st.code('\n'.join(output_lines))
                
                # Show preview images if generated
                preview_dir = run_dir / "test_preview"
                if preview_dir.exists():
                    preview_files = sorted(preview_dir.glob("*.png"))
                    if preview_files:
                        st.markdown("### 📸 Test Preview Images")
                        cols = st.columns(2)
                        for idx, img_path in enumerate(preview_files[:10]):  # Show first 10
                            with cols[idx % 2]:
                                st.image(str(img_path), caption=img_path.name, use_column_width=True)
            else:
                status_placeholder.error("❌ Test ended with an error!")
                with st.expander("Show error output", expanded=True):
                    st.code('\n'.join(output_lines))
                    
        except Exception as e:
            status_placeholder.error(f"❌ Test execution error: {e}")
            import traceback
            st.code(traceback.format_exc())
