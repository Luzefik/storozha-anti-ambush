from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def coco_json_to_yolo(coco_json_path: Path, output_labels_dir: Path) -> int:
    """COCO json -> one YOLO .txt per image (empty = no drone); return files written.

    COCO bbox: [x_min, y_min, w, h] px. YOLO line: "cls x_c y_c w_n h_n", divided by W, H.
    """
    coco = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))

    # coco category id -> yolo class (1 -> 0)
    cat_ids = sorted(cat["id"] for cat in coco["categories"])
    cat_mapping = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    # image_id -> [[class_id, x_min, y_min, w, h], ...]
    image_annotations: dict[int, list[list[float]]] = {}
    for ann in coco["annotations"]:
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"{coco_json_path}: annotation {ann.get('id')} has bad bbox {bbox}")
        image_annotations.setdefault(ann["image_id"], []).append(
            [cat_mapping[ann["category_id"]], *bbox]
        )

    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # all images, not only annotated: no box -> empty .txt
    for img in coco["images"]:
        img_w, img_h = img["width"], img["height"]  # per image, videos differ in size
        out_file = output_labels_dir / f"{Path(img['file_name']).stem}.txt"

        anns = image_annotations.get(img["id"])
        if not anns:
            out_file.write_text("")  # negative
            continue

        np_anns = np.array(anns, dtype=np.float64)  # (K, 5)
        class_ids = np_anns[:, 0].astype(np.int32)  # (K,)
        x_min, y_min, w, h = np_anns[:, 1], np_anns[:, 2], np_anns[:, 3], np_anns[:, 4]

        # clip corners in px, before normalising: frame edges are exactly 0..W, 0..H
        # (clipping x_c, w_n alone can't help: x_c=1.0, w_n=0.03 are in [0, 1] but stick out)
        # 9 boxes in video-2 / video-9 stick out by up to 3.6 px
        x1 = np.clip(x_min, 0.0, img_w)
        y1 = np.clip(y_min, 0.0, img_h)
        x2 = np.clip(x_min + w, 0.0, img_w)
        y2 = np.clip(y_min + h, 0.0, img_h)

        # fully outside -> 0 px wide after clip, drop it; one mask keeps rows aligned
        keep = (x2 > x1) & (y2 > y1)
        if not keep.any():
            out_file.write_text("")  # nothing left -> negative
            continue
        class_ids, x1, y1, x2, y2 = class_ids[keep], x1[keep], y1[keep], x2[keep], y2[keep]

        x_center = (x1 + x2) / 2.0 / img_w
        y_center = (y1 + y2) / 2.0 / img_h
        w_n = (x2 - x1) / img_w
        h_n = (y2 - y1) / img_h

        yolo_array = np.stack([class_ids, x_center, y_center, w_n, h_n], axis=1)  # (K, 5)
        np.savetxt(out_file, yolo_array, fmt="%d %.6f %.6f %.6f %.6f")

    return len(coco["images"])
