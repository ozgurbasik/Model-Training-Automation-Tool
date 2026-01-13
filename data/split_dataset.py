import argparse
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def gather_images(dataset_root: Path) -> List[Path]:
    image_paths: List[Path] = []
    for split_dir in dataset_root.iterdir():
        if not split_dir.is_dir():
            continue
        if split_dir.name.endswith("_Annotations"):
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(split_dir.glob(ext))
    return sorted(image_paths)


def write_split(
    image_paths: List[Path], train_file: Path, val_file: Path
) -> Tuple[int, int]:
    train_paths, val_paths = train_test_split(
        image_paths, test_size=0.2, random_state=42, shuffle=True
    )
    train_file.parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, "w", encoding="utf-8") as f:
        for p in train_paths:
            f.write(str(p.as_posix()) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for p in val_paths:
            f.write(str(p.as_posix()) + "\n")
    return len(train_paths), len(val_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val split file lists from image folders."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="Dataset",
        help="Root folder containing image subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="splits",
        help="Output directory for train.txt and val.txt.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_paths = gather_images(dataset_root)
    if not image_paths:
        raise SystemExit(f"No images found under {dataset_root}")

    output_dir = Path(args.output_dir)
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    n_train, n_val = write_split(image_paths, train_file, val_file)
    print(f"Wrote {n_train} train and {n_val} val image paths.")


if __name__ == "__main__":
    main()


