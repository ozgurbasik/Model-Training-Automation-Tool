import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path for local imports when launched outside repo root
# __file__ = app/tabs/validation_tab.py -> parent.parent.parent = repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_project_root_on_path() -> None:
    """Guarantee project root is importable (Streamlit may change CWD)."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def render_dataset_validation_tab() -> None:
    """Render the dataset validation tab."""

    st.subheader("Dataset Format Check")
    st.info("Preferred format: Label Studio annotations.")
    st.info("Use the 'Dataset Transformation' tab for conversions.")

    dataset_root = st.text_input("Dataset root directory", value="Dataset", key="validation_dataset_root")
    st.info("Enter the full path (e.g., C:/Users/.../RC-Car-Model-Training/DataSet/RcCArDataset)")

    st.divider()

    check_labelme = st.checkbox("Check Uniformed JSON Format", value=True)
    check_detection = st.checkbox("Check Detection (YOLO) format", value=False)
    check_segmentation = st.checkbox("Check Segmentation format", value=False)

    detection_labels_root = None
    segmentation_masks_root = None

    if check_detection:
        detection_labels_root = st.text_input("Detection labels directory", value="labels/detection")

    if check_segmentation:
        segmentation_masks_root = st.text_input("Segmentation masks directory", value="labels/segmentation")

    validation_key = f"validation_{dataset_root}_{check_labelme}_{check_detection}_{check_segmentation}"

    if st.button("Validate Dataset"):
        with st.spinner("Validating dataset..."):
            _ensure_project_root_on_path()

            try:
                project_root = Path(__file__).parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                original_cwd = os.getcwd()
                try:
                    os.chdir(project_root)
                    from data.validate_dataset import validate_dataset
                finally:
                    os.chdir(original_cwd)

                results = validate_dataset(
                    Path(dataset_root),
                    check_labelme=check_labelme,
                    check_detection=check_detection,
                    check_segmentation=check_segmentation,
                    detection_labels_root=detection_labels_root,
                    segmentation_masks_root=segmentation_masks_root,
                )

                st.session_state[validation_key] = results
                st.session_state["validation_dataset_root_value"] = dataset_root
                if f"viz_files_{dataset_root}" in st.session_state:
                    del st.session_state[f"viz_files_{dataset_root}"]

            except ImportError as e:
                st.error(f"Import error: {e}")
                st.info("Please ensure you run Streamlit from the project root.")
                st.code(f"Project root: {project_root}")
                return
            except Exception as e:  # noqa: F841
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())
                return

    if validation_key in st.session_state and st.session_state.get("validation_dataset_root_value") == dataset_root:
        results = st.session_state[validation_key]
    elif validation_key not in st.session_state:
        st.info("Please click the 'Validate Dataset' button.")
        return
    else:
        st.info("Dataset root directory changed. Please click 'Validate Dataset' again.")
        return

    st.markdown("### Results")

    stats = results.get("statistics", {})
    if stats:
        st.markdown("#### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", stats.get("total_images", "N/A"))
        with col2:
            st.metric("Total Labels", stats.get("total_labels", "N/A"))
        with col3:
            st.metric("Images without Labels", stats.get("images_without_labels", 0))
        with col4:
            st.metric("Labels without Images", stats.get("labels_without_images", 0))

    if check_labelme:
        st.markdown("#### LabelMe Format")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total files", results["labelme"]["total"])
        with col2:
            st.metric("Valid", results["labelme"]["valid"]) 
        with col3:
            st.metric("Errors", len(results["labelme"]["errors"]))
        with col4:
            st.metric("Warnings", len(results["labelme"].get("warnings", [])))

        if stats.get("images_without_labels", 0) > 0:
            st.error(f"⚠️ Labels not found for {stats['images_without_labels']} images")
            with st.expander("Images without labels (show all)", expanded=True):
                images_without_labels = stats.get("images_without_labels_list", [])
                if images_without_labels:
                    num_cols = 4
                    cols = st.columns(num_cols)
                    for idx, img_name in enumerate(images_without_labels):
                        with cols[idx % num_cols]:
                            st.text(img_name)
                else:
                    st.text("Details may be available in warning messages")

        if stats.get("labels_without_images", 0) > 0:
            st.error(f"⚠️ Images not found for {stats['labels_without_images']} labels")
            with st.expander("Labels without corresponding images (show all)", expanded=True):
                labels_without_images = stats.get("labels_without_images_list", [])
                if labels_without_images:
                    num_cols = 4
                    cols = st.columns(num_cols)
                    for idx, label_name in enumerate(labels_without_images):
                        with cols[idx % num_cols]:
                            st.text(label_name)
                else:
                    st.text("Details may be available in warning messages")

        if results["labelme"]["errors"]:
            st.error("Errors found:")
            with st.expander("Error details", expanded=False):
                for err in results["labelme"]["errors"][:20]:
                    st.text(err)
                if len(results["labelme"]["errors"]) > 20:
                    st.text(f"... and {len(results['labelme']['errors']) - 20} more errors")

        if results["labelme"].get("warnings"):
            st.warning("Warnings:")
            with st.expander("All warning details", expanded=False):
                for warn in results["labelme"]["warnings"]:
                    st.text(warn)

    # Cleanup action: remove images without labels and labels without images
    # Import and setup outside cleanup section so it's available during button click
    try:
        from data.validate_dataset import find_annotation_image_pairs
        ann_dirs, img_dirs = find_annotation_image_pairs(Path(dataset_root))
    except Exception as cleanup_import_err:
        ann_dirs, img_dirs = [], []

    img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"]
    label_exts = [".json"]
    
    def _search_by_stem(stem: str, dirs, exts):
        for d in dirs:
            if not d.exists():
                continue
            for ext in exts:
                candidate = next(d.rglob(f"{stem}{ext}"), None)
                if candidate and candidate.exists():
                    return candidate
        return None

    img_wo_labels = stats.get("images_without_labels_list", []) if stats else []
    labels_wo_imgs = stats.get("labels_without_images_list", []) if stats else []
    if img_wo_labels or labels_wo_imgs:
        st.markdown("### Cleanup")
        st.info("This will delete images without labels and label files without images from the selected directory.")

        paths = results.get("paths", {}) if results else {}
        image_map = {k: Path(v) for k, v in paths.get("images", {}).items()}
        label_map = {k: Path(v) for k, v in paths.get("labels", {}).items()}

        with st.expander("Preview files to be deleted", expanded=False):
            if img_wo_labels:
                st.write("Images without labels:")
                for stem in img_wo_labels:
                    path = image_map.get(stem) or _search_by_stem(stem, img_dirs, img_exts)
                    st.text(str(path if path else f"(path not found) {stem}"))
            if labels_wo_imgs:
                st.write("Labels without corresponding images:")
                for stem in labels_wo_imgs:
                    path = label_map.get(stem) or _search_by_stem(stem, ann_dirs, label_exts)
                    st.text(str(path if path else f"(path not found) {stem}"))

        if st.button("Delete images without labels and labels without images"):
            deleted_images = 0
            deleted_labels = 0
            not_found = []

            for stem in img_wo_labels:
                path = image_map.get(stem) or _search_by_stem(stem, img_dirs, img_exts)
                if not path:
                    not_found.append(f"(image) {stem}")
                    continue
                if path.exists():
                    try:
                        path.unlink()
                        deleted_images += 1
                    except Exception:
                        st.warning(f"Could not delete (image): {path}")
                else:
                    not_found.append(f"(image, missing) {stem} → {path}")

            for stem in labels_wo_imgs:
                path = label_map.get(stem) or _search_by_stem(stem, ann_dirs, label_exts)
                if not path:
                    not_found.append(f"(label) {stem}")
                    continue
                if path.exists():
                    try:
                        path.unlink()
                        deleted_labels += 1
                    except Exception:
                        st.warning(f"Could not delete (label): {path}")
                else:
                    not_found.append(f"(label, missing) {stem} → {path}")

            st.success(f"Deletion complete. Deleted images: {deleted_images}, Deleted labels: {deleted_labels}.")
            if not_found:
                st.warning(f"Not found/Couldn't delete: {len(not_found)} file(s)")
                with st.expander("Details", expanded=False):
                    for item in not_found:
                        st.text(item)
            st.info("Click 'Validate Dataset' again to refresh statistics.")

    st.markdown("### Sample Visualization")
    if check_labelme and Path(dataset_root).exists():
        viz_cache_key = f"viz_files_{dataset_root}"
        
        # Add cache clear button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Clear Cache", key="clear_viz_cache"):
                if viz_cache_key in st.session_state:
                    del st.session_state[viz_cache_key]
                st.success("Cache cleared!")
                st.rerun()

        def _load_labelme_data(json_path: Path):
            """Load JSON data without conversion - work with native format."""
            with open(json_path, "r", encoding="utf-8") as f:
                data_local = json.load(f)
            return data_local

        if viz_cache_key not in st.session_state:
            from data.validate_dataset import find_annotation_image_pairs

            ann_dirs, img_dirs = find_annotation_image_pairs(Path(dataset_root))

            all_json_files = []
            for ann_dir in ann_dirs:
                json_files = sorted(ann_dir.glob("*.json"))
                for subdir in ann_dir.iterdir():
                    if subdir.is_dir():
                        json_files.extend(sorted(subdir.glob("*.json")))
                all_json_files.extend(json_files)

            json_files_with_shapes = []
            for json_file in all_json_files:
                try:
                    data = _load_labelme_data(json_file)
                    # Check for shapes (old format)
                    has_annotations = False
                    if data.get("shapes"):
                        for shape in data.get("shapes", []):
                            points = shape.get("points", [])
                            if len(points) >= 2:  # Support both bbox (2 pts) and polygon (3+ pts)
                                has_annotations = True
                                break
                    # Check for labels (uniform format)
                    if not has_annotations and data.get("labels"):
                        for lbl in data.get("labels", []):
                            points = lbl.get("points", [])
                            if len(points) >= 2:  # Support both bbox (2 pts) and polygon (3+ pts)
                                has_annotations = True
                                break
                    
                    if has_annotations:
                        json_files_with_shapes.append(json_file)
                except Exception:
                    pass

            st.session_state[viz_cache_key] = {
                "json_files": json_files_with_shapes,
                "ann_dirs": ann_dirs,
                "img_dirs": img_dirs,
            }
            st.session_state.sample_viz_idx = 0

        cached_data = st.session_state[viz_cache_key]
        json_files_with_shapes = cached_data["json_files"]
        ann_dirs = cached_data["ann_dirs"]
        img_dirs = cached_data["img_dirs"]

        if ann_dirs and img_dirs and json_files_with_shapes:
            max_idx = len(json_files_with_shapes) - 1

            if st.session_state.sample_viz_idx >= len(json_files_with_shapes):
                st.session_state.sample_viz_idx = 0

            sample_idx = st.slider(
                "Select sample image", 0, max_idx, value=st.session_state.sample_viz_idx, key="sample_viz_slider"
            )
            st.session_state.sample_viz_idx = sample_idx

            if sample_idx < len(json_files_with_shapes):
                json_path = json_files_with_shapes[sample_idx]
                try:
                    data = _load_labelme_data(json_path)

                    img_dir = None
                    for ann_dir, img_dir_candidate in zip(ann_dirs, img_dirs):
                        if json_path.parent == ann_dir or (ann_dir.name == "labels_all" and json_path.parent == ann_dir.parent):
                            img_dir = img_dir_candidate
                            break

                    if img_dir is None and img_dirs:
                        img_dir = img_dirs[0]

                    if img_dir:
                        json_stem = json_path.stem
                        img_path = None

                        subdirs = [item for item in img_dir.iterdir() if item.is_dir()]
                        if subdirs:
                            if "_" in json_stem:
                                prefix = json_stem.split("_")[0]
                                for subdir in subdirs:
                                    if subdir.name == prefix:
                                        for ext in [".png", ".jpg", ".jpeg"]:
                                            candidate = subdir / f"{json_stem}{ext}"
                                            if candidate.exists():
                                                img_path = candidate
                                                break
                                        break

                            if img_path is None:
                                for subdir in subdirs:
                                    for ext in [".png", ".jpg", ".jpeg"]:
                                        candidate = subdir / f"{json_stem}{ext}"
                                        if candidate.exists():
                                            img_path = candidate
                                            break
                                    if img_path:
                                        break
                        else:
                            for ext in [".png", ".jpg", ".jpeg"]:
                                candidate = img_dir / f"{json_stem}{ext}"
                                if candidate.exists():
                                    img_path = candidate
                                    break

                        if img_path and img_path.exists():
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                actual_height, actual_width = img_rgb.shape[:2]



                                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                                ax.imshow(img_rgb)

                                # Handle old format (shapes)
                                for shape in data.get("shapes", []):
                                    label = shape.get("label", "")
                                    points = shape.get("points", [])
                                    if points:
                                        points_array = np.array(points, dtype=np.float32)
                                        
                                        # Bounding box (2 points: [[x_min, y_min], [x_max, y_max]])
                                        if len(points_array) == 2:
                                            x_min, y_min = points_array[0]
                                            x_max, y_max = points_array[1]
                                            # Draw rectangle
                                            from matplotlib.patches import Rectangle
                                            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                           linewidth=2, edgecolor='red', facecolor='none')
                                            ax.add_patch(rect)
                                            # Add label
                                            ax.text(x_min, y_min - 5, label, color="red", fontsize=12,
                                                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                                        
                                        # Polygon (3+ points)
                                        elif len(points_array) >= 3:
                                            ax.plot(
                                                np.append(points_array[:, 0], points_array[0, 0]),
                                                np.append(points_array[:, 1], points_array[0, 1]),
                                                "r-",
                                                linewidth=2,
                                            )
                                            if points_array.size > 0:
                                                ax.text(
                                                    points_array[0, 0],
                                                    points_array[0, 1] - 5,
                                                    label,
                                                    color="red",
                                                    fontsize=12,
                                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                                                )

                                # Handle uniform format (labels)
                                for lbl in data.get("labels", []):
                                    pts = lbl.get("points", [])
                                    lbls = lbl.get("polygonlabels", [])
                                    label = lbls[0] if lbls else ""
                                    if not pts or len(pts) < 2:  # Support both bbox (2 pts) and polygon (3+ pts)
                                        continue
                                    
                                    # Get coordinate type and original dimensions
                                    coord_type = lbl.get("coordinate_type", "normalized")
                                    orig_width = lbl.get("original_width", actual_width)
                                    orig_height = lbl.get("original_height", actual_height)
                                    
                                    # Convert coordinates based on type and scale to actual image size
                                    if coord_type == "actual":
                                        # Coordinates are in pixels for original image - use directly if dimensions match
                                        if orig_width == actual_width and orig_height == actual_height:
                                            # No scaling needed - dimensions match
                                            points_array = np.array(pts, dtype=np.float32)
                                        else:
                                            # Scale to current image size
                                            scale_x = actual_width / orig_width if orig_width else 1.0
                                            scale_y = actual_height / orig_height if orig_height else 1.0
                                            points_array = np.array([[p[0] * scale_x, p[1] * scale_y] for p in pts], dtype=np.float32)
                                    else:
                                        # Normalized coordinates (0-100 range) - convert to actual image pixels
                                        points_array = np.array([[p[0] / 100.0 * actual_width, p[1] / 100.0 * actual_height] for p in pts], dtype=np.float32)
                                    
                                    # Bounding box (2 points: [[x_min, y_min], [x_max, y_max]])
                                    if len(points_array) == 2:
                                        x_min, y_min = points_array[0]
                                        x_max, y_max = points_array[1]
                                        # Draw rectangle
                                        from matplotlib.patches import Rectangle
                                        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                       linewidth=2, edgecolor='red', facecolor='none')
                                        ax.add_patch(rect)
                                        # Add label
                                        ax.text(x_min, y_min - 5, label, color="red", fontsize=12,
                                               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                                    
                                    # Polygon (3+ points)
                                    elif len(points_array) >= 3:
                                        ax.plot(
                                            np.append(points_array[:, 0], points_array[0, 0]),
                                            np.append(points_array[:, 1], points_array[0, 1]),
                                            "r-",
                                            linewidth=2,
                                        )
                                        if points_array.size > 0:
                                            ax.text(
                                                points_array[0, 0],
                                                points_array[0, 1] - 5,
                                                label,
                                                color="red",
                                                fontsize=12,
                                                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                                            )

                                ax.set_title(f"Sample: {json_path.name}")
                                ax.axis("off")
                                st.pyplot(fig)
                            else:
                                st.warning(f"Image could not be read: {img_path}")
                        else:
                            st.warning(f"Image not found: {json_path.stem}")
                except Exception as e:  # noqa: F841
                    st.error(f"Visualization error: {e}")
                    st.code(traceback.format_exc())
        else:
            st.info("No annotations to visualize (all files empty)")
