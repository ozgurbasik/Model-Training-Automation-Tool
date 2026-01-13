import importlib
import json
import sys
import traceback
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Polygon

# Ensure project root is on sys.path for local imports when launched outside repo root
# __file__ = app/tabs/dataset_transformation_tab.py -> parent.parent.parent = repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_project_root_on_path() -> None:
    """Guarantee project root is importable (Streamlit may change CWD)."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def _draw_shape_overlays(ax, data: dict, img_height: int, img_width: int) -> None:
    """Overlay polygons from LabelMe/Label Studio data (filled + outline)."""
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        points = shape.get("points", [])
        if not points or len(points) < 3:
            continue
        pts_array = np.array(points, dtype=np.float32)
        poly = Polygon(pts_array, closed=True, facecolor="red", alpha=0.25, edgecolor="red", linewidth=2)
        ax.add_patch(poly)
        ax.text(
            pts_array[0, 0],
            pts_array[0, 1] - 10,
            label,
            color="red",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Handle Label Studio style polygons if present
    for lbl in data.get("labels", []):
        pts = lbl.get("points", [])
        lbls = lbl.get("polygonlabels", [])
        label = lbls[0] if lbls else ""
        if not pts or len(pts) < 3:
            continue
        
        # Check coordinate type - respect actual vs normalized coordinates
        coord_type = lbl.get("coordinate_type", "normalized")
        
        if coord_type == "actual":
            # Coordinates are already in pixels - use directly
            abs_pts = np.array(pts, dtype=np.float32)
        else:
            # Normalized coordinates (0-100 range) - convert to pixels
            abs_pts = np.array([[p[0] / 100.0 * img_width, p[1] / 100.0 * img_height] for p in pts], dtype=np.float32)
        
        poly = Polygon(abs_pts, closed=True, facecolor="red", alpha=0.25, edgecolor="red", linewidth=2)
        ax.add_patch(poly)
        ax.text(
            abs_pts[0, 0],
            abs_pts[0, 1] - 10,
            label,
            color="red",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )


def render_dataset_transformation_tab() -> None:
    """Render dataset transformation utilities tab."""

    st.subheader("Dataset Transformations")
    st.info("Perform LabelMe → Label Studio conversion, H1/H1_Annotations merge, and vertical image rotation here.")

    dataset_root = st.text_input("Dataset root directory", value="Dataset", key="transform_dataset_root")
    st.info("Enter the full path (e.g., C:/Users/.../RC-Car-Model-Training/DataSet/RcCArDataset)")

    st.divider()

    # ========== H1/H1_Annotations Reorganization ==========
    st.markdown("### 📁 Dataset Reorganization")
    st.info("Converts H1/H1_Annotations format to images_all/labels_all format.")

    if st.button("H1/H1_Annotations → images_all/labels_all Conversion"):
        if not Path(dataset_root).exists():
            st.error(f"Dataset directory not found: {dataset_root}")
        else:
            with st.spinner("Reorganizing dataset..."):
                try:
                    _ensure_project_root_on_path()

                    from data.reorganize_to_merged import reorganize_dataset

                    reorganize_dataset(Path(dataset_root), dry_run=False)
                    st.success("✅ Dataset reorganized successfully!")
                    st.info("Original H1, H1_Annotations, ... folders were deleted. Only images_all/ and labels_all/ remain.")
                except Exception as e:  # noqa: F841
                    st.error(f"Reorganization error: {e}")
                    st.code(traceback.format_exc())

    st.info("💡 This converts H1/H1_Annotations to images_all/labels_all and renames files with prefixes like H1_frame_001. Original folders are deleted after the process.")

    st.divider()

    # ========== BUTTON 1: Uniform All JSON Files ==========
    st.markdown("### 🔄 Unify JSON Format (Uniform)")
    st.info("Convert all JSON files to a single format. Uses Format 2 style with actual coordinates.")
    
    create_backup_uniform = st.checkbox("Create .backup during conversion (Uniform)", value=True, key="backup_uniform")
    
    if st.button("Convert All JSONs to Uniform Format (Actual Coordinates)"):
        if not Path(dataset_root).exists():
            st.error(f"Dataset directory not found: {dataset_root}")
        else:
            with st.spinner("Converting JSON files to uniform format..."):
                try:
                    _ensure_project_root_on_path()
                    from data.uniform_format_converter import uniform_all_jsons
                    
                    stats = uniform_all_jsons(Path(dataset_root), backup=create_backup_uniform)
                    
                    st.success(f"✅ {stats['converted']} files converted to uniform format!")
                    st.info(
                        f"Total: {stats['total']} | Converted: {stats['converted']} | Already uniform: {stats['already_uniform']}"
                    )
                    
                    # Show format breakdown
                    with st.expander("Format Breakdown", expanded=True):
                        st.write("Original formats:")
                        for fmt, count in stats['by_format'].items():
                            if count > 0:
                                st.write(f"- {fmt}: {count} file(s)")
                    
                    if stats["errors"]:
                        st.warning(f"⚠️ {len(stats['errors'])} files could not be converted.")
                        with st.expander("Error details", expanded=False):
                            for err in stats["errors"][:20]:
                                st.text(err)
                            if len(stats["errors"]) > 20:
                                st.text(f"... and {len(stats['errors']) - 20} more errors")
                
                except Exception as e:
                    st.error(f"Conversion error: {e}")
                    st.code(traceback.format_exc())
    
    st.markdown("""
    **Uniform Format Features:**
    - Structure in Format 2 (Label Studio) style
    - Uses the `labels` array
    - Actual pixel coordinates
    - Tracked via `coordinate_type`
    - Always stores `original_width` and `original_height`
    """)
    
    st.divider()
    
    # ========== BUTTON 2: Detect and Convert Coordinates ==========
    st.markdown("### 🔍 Coordinate Type Detection and Conversion")
    st.info("Detect coordinate types in uniform JSONs and perform normalized ↔ actual conversions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Analyze Coordinate Types"):
            if not Path(dataset_root).exists():
                st.error(f"Dataset directory not found: {dataset_root}")
            else:
                with st.spinner("Analyzing coordinate types..."):
                    try:
                        _ensure_project_root_on_path()
                        from data.uniform_format_converter import analyze_dataset_coordinates
                        
                        analysis = analyze_dataset_coordinates(Path(dataset_root))
                        
                        st.markdown("#### 📊 Analysis Results")
                        
                        # Show statistics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Actual Coordinates", analysis['actual'])
                        with col_b:
                            st.metric("Normalized Coordinates", analysis['normalized'])
                        with col_c:
                            st.metric("Unknown/Mixed", analysis['unknown'] + analysis['mixed'])
                        
                        st.info(f"Total analyzed: {analysis['total']} files")
                        
                        # Show samples
                        if analysis['samples']:
                            with st.expander("Sample Files", expanded=True):
                                for sample in analysis['samples']:
                                    st.write(f"**{sample['file']}**: {sample['type']}")
                                    if 'sample_points' in sample['info']:
                                        st.write(f"  Sample points: {sample['info']['sample_points'][:2]}")
                        
                        if analysis['errors']:
                            with st.expander("Errors", expanded=False):
                                for err in analysis['errors'][:10]:
                                    st.text(err)
                    
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        st.code(traceback.format_exc())
    
    with col2:
        st.markdown("**Perform Conversion:**")
        conversion_type = st.radio(
            "Target format",
            ["Convert to Actual Coordinates", "Convert to Normalized (0-100)"],
            key="conversion_radio"
        )
        
        create_backup_coords = st.checkbox("Create backup", value=True, key="backup_coords")
        
        if st.button("Convert Coordinates"):
            target = "actual" if "Actual" in conversion_type else "normalized"
            
            if not Path(dataset_root).exists():
                st.error(f"Dataset directory not found: {dataset_root}")
            else:
                with st.spinner(f"Converting coordinates to {target} format..."):
                    try:
                        _ensure_project_root_on_path()
                        from data.uniform_format_converter import batch_convert_coordinates
                        
                        stats = batch_convert_coordinates(
                            Path(dataset_root),
                            target_type=target,
                            backup=create_backup_coords
                        )
                        
                        st.success(f"✅ {stats['converted']} files converted to {target} format!")
                        st.info(
                            f"Total: {stats['total']} | Converted: {stats['converted']} | Already in target format: {stats['already_target']}"
                        )
                        
                        if stats['errors']:
                            st.warning(f"⚠️ {len(stats['errors'])} files could not be converted.")
                            with st.expander("Error details", expanded=False):
                                for err in stats['errors'][:20]:
                                    st.text(err)
                    
                    except Exception as e:
                        st.error(f"Conversion error: {e}")
                        st.code(traceback.format_exc())
    
    st.markdown("""
    **Coordinate Types:**
    - **Actual**: Coordinates in pixels (e.g., x=245, y=180)
    - **Normalized**: Coordinates in percentage (e.g., x=28.89, y=37.66, range 0-100)
    """)
    
    st.divider()
    
    # ========== BUTTON 3: Polygon to Bounding Box Conversion ==========
    st.markdown("### 📦 Dataset Transformation: Segmentation (Polygon) → Object Detection (Bounding Box)")
    st.warning("⚠️ **WARNING**: This operation is IRREVERSIBLE! Polygon coordinates will be converted to bounding boxes and original polygon information will be lost.")
    st.info("Converts segmentation polygon masks into bounding box format for object detection models.")
    
    create_backup_bbox = st.checkbox("Create .backup during conversion (Polygon→BBox)", value=True, key="backup_bbox")
    
    col_preview, col_convert = st.columns([1, 1])
    
    with col_preview:
        if st.button("🔍 Preview: How Many Files Will Be Affected?"):
            if not Path(dataset_root).exists():
                st.error(f"Dataset directory not found: {dataset_root}")
            else:
                with st.spinner("Analyzing files..."):
                    try:
                        _ensure_project_root_on_path()
                        labels_dir = Path(dataset_root) / "labels_all"
                        
                        if not labels_dir.exists():
                            st.error(f"labels_all directory not found: {labels_dir}")
                        else:
                            json_files = list(labels_dir.glob("*.json"))
                            polygon_count = 0
                            file_count = 0
                            
                            for json_file in json_files:
                                try:
                                    with open(json_file, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    
                                    # Check for polygons in shapes
                                    shapes = data.get("shapes", [])
                                    labels = data.get("labels", [])
                                    
                                    has_polygon = False
                                    for shape in shapes:
                                        points = shape.get("points", [])
                                        if len(points) >= 3:
                                            polygon_count += 1
                                            has_polygon = True
                                    
                                    for lbl in labels:
                                        points = lbl.get("points", [])
                                        if len(points) >= 3:
                                            polygon_count += 1
                                            has_polygon = True
                                    
                                    if has_polygon:
                                        file_count += 1
                                
                                except Exception:
                                    continue
                            
                            st.success(f"✅ Analysis complete!")
                            st.metric("Number of Files Affected", file_count)
                            st.metric("Total Polygon Count", polygon_count)
                            st.info(f"Total JSON files: {len(json_files)}")
                    
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        st.code(traceback.format_exc())
    
    with col_convert:
        if st.button("🔄 Convert Polygon → Bounding Box"):
            if not Path(dataset_root).exists():
                st.error(f"Dataset directory not found: {dataset_root}")
            else:
                with st.spinner("Converting polygons to bounding boxes..."):
                    try:
                        _ensure_project_root_on_path()
                        from data.uniform_format_converter import convert_polygons_to_bboxes
                        
                        stats = convert_polygons_to_bboxes(Path(dataset_root), backup=create_backup_bbox)
                        
                        st.success(f"✅ {stats['converted_files']} files converted successfully!")
                        st.info(
                            f"Total: {stats['total']} file(s) | "
                            f"Converted: {stats['converted_files']} file(s) | "
                            f"Polygon→BBox: {stats['converted_polygons']} count"
                        )
                        
                        if stats['skipped']:
                            st.info(f"⏭️ {stats['skipped']} file(s) are already in bounding box format or contain no polygons.")
                        
                        if stats['errors']:
                            st.warning(f"⚠️ {len(stats['errors'])} files could not be converted.")
                            with st.expander("Error details", expanded=False):
                                for err in stats['errors'][:20]:
                                    st.text(err)
                                if len(stats['errors']) > 20:
                                    st.text(f"... and {len(stats['errors']) - 20} more errors")
                        
                        # Automatically generate YOLO labels
                        st.divider()
                        st.info("🎯 Generating YOLO format labels automatically...")
                        
                        try:
                            from data.labelme_to_detection import (
                                find_annotation_image_pairs,
                                collect_label_map,
                                process_annotation
                            )
                            
                            dataset_path = Path(dataset_root)

                            # Default output root; will be overridden if labels_all is present
                            output_root = dataset_path / "labels"

                            # Check for labels_all and images_all directories
                            labels_all = dataset_path / "labels_all"
                            images_all = dataset_path / "images_all"

                            if labels_all.is_dir() and images_all.is_dir():
                                # Align YOLO labels with Ultralytics expectation: images_all -> labels_all
                                ann_dirs = [labels_all]
                                img_dirs = [images_all]
                                output_root = labels_all
                            else:
                                ann_dirs, img_dirs = find_annotation_image_pairs(dataset_path)
                            
                            if not ann_dirs:
                                st.warning(f"⚠️ Annotation directories not found. Dataset path: {dataset_path}")
                                st.warning(f"labels_all exists? {labels_all.exists() if labels_all else 'N/A'}")
                                st.warning(f"images_all exists? {images_all.exists() if images_all else 'N/A'}")
                            else:
                                # Debug: check what's in the annotation directory
                                st.write(f"📁 Annotation directory: {ann_dirs[0]}")
                                json_count = len(list(ann_dirs[0].glob("*.json")))
                                st.write(f"📄 Number of JSON files: {json_count}")
                                
                                # Sample a file for debugging
                                sample_json = next(ann_dirs[0].glob("*.json"), None)
                                if sample_json:
                                    with open(sample_json, "r", encoding="utf-8") as f:
                                        sample_data = json.load(f)
                                    st.write(f"Sample file: {sample_json.name}")
                                    st.write(f"  - Has 'shapes'? {len(sample_data.get('shapes', []))}")
                                    st.write(f"  - Has 'labels'? {len(sample_data.get('labels', []))}")
                                    if sample_data.get('labels'):
                                        first_label = sample_data['labels'][0]
                                        st.write(f"  - First label: {first_label.get('polygonlabels', 'NONE')}")
                                
                                label_map = collect_label_map(ann_dirs)
                                
                                # Manual collection as fallback
                                if not label_map:
                                    st.warning("⚠️ collect_label_map() returned empty, performing manual collection...")
                                    label_map = {}
                                    for ann_dir in ann_dirs:
                                        for json_path in sorted(ann_dir.glob("*.json")):
                                            with open(json_path, "r", encoding="utf-8") as f:
                                                data = json.load(f)
                                            
                                            # Handle shapes format (LabelMe)
                                            for shape in data.get("shapes", []):
                                                label = shape.get("label")
                                                if label and label not in label_map:
                                                    label_map[label] = len(label_map)
                                            
                                            # Handle labels format (uniform)
                                            for lbl in data.get("labels", []):
                                                label_list = lbl.get("polygonlabels", [])
                                                label = label_list[0] if label_list else None
                                                if label and label not in label_map:
                                                    label_map[label] = len(label_map)
                                    
                                    st.write(f"Manual collection result: {len(label_map)} classes found")
                                
                                if not label_map:
                                    st.error("❌ No classes found! collect_label_map() returned empty.")
                                else:
                                    st.write("**Detected classes:**")
                                    for name, cid in sorted(label_map.items(), key=lambda x: x[1]):
                                        st.write(f"  {cid}: {name}")
                                    
                                    total_labels = 0
                                    total_objects = 0
                                    for ann_dir, img_dir in zip(ann_dirs, img_dirs):
                                        if output_root == labels_all:
                                            # Keep labels alongside images_all for YOLO to auto-discover
                                            split_output_dir = output_root
                                        else:
                                            split_name = ann_dir.name.replace("_Annotations", "")
                                            split_output_dir = output_root / split_name
                                        
                                        json_files = sorted(ann_dir.glob("*.json"))
                                        for json_path in json_files:
                                            # Count objects before processing
                                            with open(json_path, "r", encoding="utf-8") as f:
                                                data = json.load(f)
                                            num_objects = len(data.get("labels", [])) + len(data.get("shapes", []))
                                            total_objects += num_objects
                                            
                                            process_annotation(json_path, img_dir, split_output_dir, label_map)
                                            total_labels += 1
                                    
                                    # Save label map
                                    names_path = output_root / "classes.txt"
                                    output_root.mkdir(parents=True, exist_ok=True)
                                    with open(names_path, "w", encoding="utf-8") as f:
                                        for name, cid in sorted(label_map.items(), key=lambda x: x[1]):
                                            f.write(f"{cid} {name}\n")
                                    
                                    st.success(f"✅ YOLO labels generated: {total_labels} files, {total_objects} objects")
                                    st.info(f"📁 Labels: {output_root}")
                                    st.info(f"📋 Class list: {names_path}")
                        
                        except Exception as yolo_error:
                            st.error(f"YOLO label generation error: {yolo_error}")
                            st.code(traceback.format_exc())
                    
                    except Exception as e:
                        st.error(f"Conversion error: {e}")
                        st.code(traceback.format_exc())
    
    st.markdown("""
    **How Does Bounding Box Conversion Work?**
    - Finds min/max X and Y coordinates for each polygon
    - Creates a rectangular bounding box from these coordinates
    - Format: `[x_min, y_min, x_max, y_max]` or `[[x_min, y_min], [x_max, y_max]]`
    - Original polygon coordinates are permanently removed (BACKUP RECOMMENDED!)
    - Coordinate type (actual/normalized) is preserved
    - **YOLO format labels are automatically generated** (.txt files)
    """)
