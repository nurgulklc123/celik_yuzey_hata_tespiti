"""NEU-DET XML (Pascal VOC) etiketlerini YOLO formatına çevirir ve YOLO klasör yapısını kurar."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "data" / "NEU-DET"
IMG_DIR = SRC_DIR / "IMAGES"
ANN_DIR = SRC_DIR / "ANNOTATIONS"
SPLIT_CSV = PROJECT_DIR / "splits" / "split.csv"
OUT_DIR = PROJECT_DIR / "data" / "yolo"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def voc_to_yolo_lines(xml_path):
    """Bir XML dosyasını okur, her kutu için bir YOLO satırı döndürdum."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_id = CLASSES.index(obj.find("name").text)
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def write_data_yaml():
    """YOLO'nun veri setini bulması için data.yaml dosyasını yazdım."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    text = (
        f"path: {OUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )
    (OUT_DIR / "data.yaml").write_text(text)


def main():
    split_df = pd.read_csv(SPLIT_CSV)

    # Eski çıktı varsa sil, her seferinde temiz başla
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    box_count = 0
    for row in split_df.itertuples():
        img_dst = OUT_DIR / "images" / row.split
        lbl_dst = OUT_DIR / "labels" / row.split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        shutil.copy(IMG_DIR / row.file, img_dst / row.file)

        stem = Path(row.file).stem
        lines = voc_to_yolo_lines(ANN_DIR / f"{stem}.xml")
        (lbl_dst / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        box_count += len(lines)

    write_data_yaml()

    print("Resim:", len(split_df))
    print("Kutu:", box_count)
    print("Kaydedildi:", OUT_DIR)


if __name__ == "__main__":
    main()