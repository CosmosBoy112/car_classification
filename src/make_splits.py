"""Step 1: write the fixed train/val/test split files.

- train/val: stratified 85/15 split of the official train set (per class).
- test: the official test set, untouched.

Image paths are resolved by scanning the class folders, because one class
name contains "/" and its folder name differs ("Ram C/V" -> "Ram C-V").

Run once:  python src/make_splits.py
"""
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C  # noqa: E402

ANNO_COLS = ["filename", "x1", "y1", "x2", "y2", "class_id"]


def load_split(anno_path, subset):
    names = pd.read_csv(C.NAMES, header=None)[0].tolist()
    df = pd.read_csv(anno_path, header=None, names=ANNO_COLS)

    # filename -> folder (filenames are unique within one subset)
    folder_of = {}
    for class_dir in sorted((C.IMAGE_ROOT / subset).iterdir()):
        for img in class_dir.glob("*.jpg"):
            assert img.name not in folder_of, f"duplicate filename {img.name}"
            folder_of[img.name] = class_dir.name
    missing = set(df.filename) - set(folder_of)
    assert not missing, f"{len(missing)} annotated images not found, e.g. {list(missing)[:3]}"

    df["label"] = df.class_id - 1  # 0-based for PyTorch
    df["class_name"] = df.class_id.map(lambda i: names[i - 1])
    df["make"] = df.class_name.map(C.make_of)
    df["path"] = [f"{subset}/{folder_of[f]}/{f}" for f in df.filename]

    # The folder must match the annotated class (after the "/" -> "-" rename)
    folder_cls = df.path.str.split("/").str[1]
    assert (folder_cls == df.class_name.str.replace("/", "-")).all(), "folder/class mismatch"

    sizes = [Image.open(C.IMAGE_ROOT / p).size for p in df.path]
    df["width"], df["height"] = zip(*sizes)
    return df[["path", "label", "class_name", "make", "x1", "y1", "x2", "y2", "width", "height"]]


def main():
    C.SPLITS_DIR.mkdir(exist_ok=True)
    full_train = load_split(C.ANNO_TRAIN, "train")
    test = load_split(C.ANNO_TEST, "test")

    train, val = train_test_split(
        full_train, test_size=C.VAL_FRACTION, stratify=full_train.label, random_state=C.SEED
    )
    train = train.sort_values("path").reset_index(drop=True)
    val = val.sort_values("path").reset_index(drop=True)

    # Sanity checks
    assert len(train) + len(val) == len(full_train) == 8144
    assert len(test) == 8041
    assert set(train.path).isdisjoint(val.path)
    for df in (train, val, test):
        assert df.label.nunique() == C.NUM_CLASSES
    per_class_val = val.label.value_counts()

    train.to_csv(C.SPLITS_DIR / "train.csv", index=False)
    val.to_csv(C.SPLITS_DIR / "val.csv", index=False)
    test.to_csv(C.SPLITS_DIR / "test.csv", index=False)

    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    print(f"val images per class: min {per_class_val.min()}, median {int(np.median(per_class_val))}, "
          f"max {per_class_val.max()}")
    print(f"train images per class: min {train.label.value_counts().min()}, "
          f"max {train.label.value_counts().max()}")


if __name__ == "__main__":
    main()
