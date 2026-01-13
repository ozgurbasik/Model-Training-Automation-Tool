import shutil
from pathlib import Path

import streamlit as st
from sklearn.model_selection import train_test_split


def render_dataset_split_tab() -> None:
    """Render dataset split tab (train/val/test)."""

    st.subheader("Dataset Split (Train/Val/Test)")

    dataset_root = st.text_input("Dataset root directory", value="Dataset_Merged", key="split_dataset_root")
    st.info("Enter the full path (e.g., C:/Users/.../RC-Car-Model-Training/DataSet/RcCArDataset)")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider("Train ratio", 0.0, 1.0, 0.7, 0.05, key="train_ratio")
    with col2:
        val_ratio = st.slider("Validation ratio", 0.0, 1.0, 0.2, 0.05, key="val_ratio")
    with col3:
        test_ratio = st.slider("Test ratio", 0.0, 1.0, 0.1, 0.05, key="test_ratio")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        st.warning(f"⚠️ Sum of ratios must be 1.00 (current: {total_ratio:.2f})")
    else:
        st.success(f"✓ Sum of ratios: {total_ratio:.2f}")

    output_dir = st.text_input("Output directory", value="splits", key="split_output_dir")
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, key="split_seed")
    copy_files = st.checkbox(
        "Copy files into train/val/test under images/labels (without subfolder hierarchy)",
        value=False,
        key="split_copy",
    )

    if st.button("Split Dataset", key="split_button"):
        if not Path(dataset_root).exists():
            st.error(f"Dataset directory not found: {dataset_root}")
            return

        if abs(total_ratio - 1.0) > 0.01:
            st.error("Please adjust ratios so their sum equals 1.0!")
            return

        try:
            with st.spinner("Splitting dataset..."):
                dataset_path = Path(dataset_root)
                image_extensions = [".png", ".jpg", ".jpeg"]
                all_images_set = set()
                all_images = []

                # Check for merged format (images_all/labels_all)
                if (dataset_path / "images_all").exists():
                    img_dir = dataset_path / "images_all"
                    for ext in image_extensions:
                        all_images_set.update(img_dir.glob(f"*{ext}"))
                        for subdir in img_dir.iterdir():
                            if subdir.is_dir():
                                all_images_set.update(subdir.glob(f"*{ext}"))
                # Check for reorganized format (images/labels)
                elif (dataset_path / "images").exists():
                    img_dir = dataset_path / "images"
                    for ext in image_extensions:
                        all_images_set.update(img_dir.glob(f"*{ext}"))
                        for subdir in img_dir.iterdir():
                            if subdir.is_dir():
                                all_images_set.update(subdir.glob(f"*{ext}"))
                else:
                    for ext in image_extensions:
                        for img_path in dataset_path.rglob(f"*{ext}"):
                            parts = {p.lower() for p in img_path.parts}
                            if any(p.endswith("_annotations") for p in parts):
                                continue
                            if any(p.startswith("lbl") for p in parts):
                                continue
                            if "labels" in parts or "labels_all" in parts:
                                continue
                            all_images_set.add(img_path)

                all_images = sorted(all_images_set)

                if not all_images:
                    st.error("No images found!")
                    return

                label_pairs = []
                # Detect label directory structure
                labels_dir = None
                if (dataset_path / "labels_all").exists():
                    labels_dir = dataset_path / "labels_all"
                elif (dataset_path / "labels").exists():
                    labels_dir = dataset_path / "labels"

                for img_path in all_images:
                    img_stem = img_path.stem
                    label_path = None
                    
                    if labels_dir:
                        label_path = labels_dir / f"{img_stem}.json"
                        if not label_path.exists():
                            label_path = labels_dir / f"{img_stem}.txt"
                    else:
                        candidates = []
                        if img_path.parent.parent:
                            candidates.append(img_path.parent.parent / f"{img_path.parent.name}_Annotations")
                            candidates.append(img_path.parent.parent / f"lbl{img_path.parent.name.lower()}")
                        if img_path.parent:
                            candidates.append(img_path.parent / f"{img_path.parent.name}_Annotations")
                            candidates.append(img_path.parent / f"lbl{img_path.parent.name.lower()}")

                        for cand_dir in candidates:
                            if not cand_dir.exists():
                                continue
                            cand = cand_dir / f"{img_stem}.json"
                            if cand.exists():
                                label_path = cand
                                break
                            cand = cand_dir / f"{img_stem}.txt"
                            if cand.exists():
                                label_path = cand
                                break

                    if label_path and label_path.exists():
                        label_pairs.append((img_path, label_path))
                    else:
                        label_pairs.append((img_path, None))

                train_val, test = train_test_split(label_pairs, test_size=test_ratio, random_state=int(random_seed))
                train, val = train_test_split(
                    train_val, test_size=val_ratio / (1 - test_ratio), random_state=int(random_seed)
                )

                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                def write_split_file(split_name, pairs):
                    split_file = output_path / f"{split_name}.txt"
                    with open(split_file, "w", encoding="utf-8") as f:
                        for img_path, _ in pairs:
                            # Write relative path from dataset_path
                            rel_path = img_path.relative_to(dataset_path)
                            f.write(f"{rel_path.as_posix()}\n")
                    return len(pairs)

                n_train = write_split_file("train", train)
                n_val = write_split_file("val", val)
                n_test = write_split_file("test", test)

                if copy_files:
                    for split_name, pairs in [("train", train), ("val", val), ("test", test)]:
                        split_dir = output_path / split_name

                        images_dir = split_dir / "images"
                        labels_dir_out = split_dir / "labels"
                        images_dir.mkdir(parents=True, exist_ok=True)
                        labels_dir_out.mkdir(parents=True, exist_ok=True)

                        for img_path, label_path in pairs:
                            # Preserve original filenames to keep image/label alignment
                            dst_img = images_dir / img_path.name
                            shutil.copy2(img_path, dst_img)

                            if label_path and label_path.exists():
                                dst_label = labels_dir_out / f"{img_path.stem}{label_path.suffix}"
                                shutil.copy2(label_path, dst_label)

                st.success("✅ Dataset split successfully!")

                st.markdown("### Split Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train", n_train, f"{n_train/len(label_pairs)*100:.1f}%")
                with col2:
                    st.metric("Validation", n_val, f"{n_val/len(label_pairs)*100:.1f}%")
                with col3:
                    st.metric("Test", n_test, f"{n_test/len(label_pairs)*100:.1f}%")
                with col4:
                    st.metric("Total", len(label_pairs))

                st.markdown("### Generated Files")
                if (output_path / "train.txt").exists():
                    st.info(f"📄 Split lists: `{output_dir}/train.txt`, `{output_dir}/val.txt`, `{output_dir}/test.txt`")

                if copy_files:
                    st.info(f"📁 Copied files: `{output_dir}/train/`, `{output_dir}/val/`, `{output_dir}/test/`")

        except Exception as e:  # noqa: F841
            st.error(f"Error: {e}")
            import traceback

            st.code(traceback.format_exc())
