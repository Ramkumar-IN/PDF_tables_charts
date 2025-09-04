#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ultralytics doclayout_yolo pdf2image')


# In[ ]:


import os
import numpy as np
from doclayout_yolo import YOLOv10
from ultralytics import YOLO
import cv2
from pdf2image import convert_from_path


# In[ ]:


get_ipython().system('apt-get update')
get_ipython().system('apt-get install -y poppler-utils')


# In[ ]:


get_ipython().system('ls "/content/drive/My Drive/Colab Notebooks/pdf_chart_tables_extractor/input_pdfs/"')


# In[45]:


#!/usr/bin/env python
# coding: utf-8


# ===============================
# Load models
# ===============================
model_detect = YOLOv10("models/doclayout_yolo_docstructbench_imgsz1024.pt")  # detection model
model_classify = YOLO("models/classification_chart.pt")  # classification model


# ===============================
# Config
# ===============================
input_pdf = "input_pdfs/TrendForce - Server DRAM Presentation of Aug. 2022 v2.pdf"

output_charts = "output_charts5"
output_tables = "output_tables5"
os.makedirs(output_charts, exist_ok=True)
os.makedirs(output_tables, exist_ok=True)

# Detection config (adjust indices to your model)
FIGURE_CLASS = 3
FIGURE_CAPTION_CLASS = 4
TABLE_CLASS = 5
TABLE_CAPTION_CLASS = 6
CONF_THRESHOLD = 0


# ===============================
# Helper functions
# ===============================
def iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


def is_contained(inner, outer, tol=0.95):
    """Check if inner box is mostly inside outer box."""
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer

    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    inner_area = (x2 - x1) * (y2 - y1)
    return inner_area > 0 and (inter_area / inner_area) >= tol


def filter_duplicates(crops, iou_thresh=0.7):
    """Remove duplicate/subset crops (keep larger area)."""
    keep = []
    # Sort by area (largest first)
    crops = sorted(
        crops,
        key=lambda x: (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]),
        reverse=True
    )

    for crop_img, conf, box in crops:
        if not any(
            iou(box, kept_box) > iou_thresh or is_contained(box, kept_box)
            for _, _, kept_box in keep
        ):
            keep.append((crop_img, conf, box))
    return keep


# ===============================
# PDF → Images → Detection → Save
# ===============================
pages = convert_from_path(input_pdf)

for page_idx, page in enumerate(pages):
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    # Run detection once
    results = model_detect(img, save=False)

    for r in results:
        boxes = r.boxes

        figures, fig_captions = [], []
        tables, tab_captions = [], []

        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if conf >= CONF_THRESHOLD:
                if cls == FIGURE_CLASS:
                    figures.append((xyxy, conf))
                elif cls == FIGURE_CAPTION_CLASS:
                    fig_captions.append((xyxy, conf))
                elif cls == TABLE_CLASS:
                    tables.append((xyxy, conf))
                elif cls == TABLE_CAPTION_CLASS:
                    tab_captions.append((xyxy, conf))

        # ===============================
        # Handle Figures → classify → save only charts
        # ===============================

        chart_crops = []
        for fig_box, fconf in figures:
            x1, y1, x2, y2 = fig_box
            merged_box = fig_box.copy()

            # Merge with nearby caption
            for cap_box, _ in fig_captions:
                cx1, cy1, cx2, cy2 = cap_box
                if abs(cy1 - y2) < 150 or abs(y1 - cy2) < 150:
                    merged_box = [
                        min(merged_box[0], cx1),
                        min(merged_box[1], cy1),
                        max(merged_box[2], cx2),
                        max(merged_box[3], cy2),
                    ]

            # Crop merged region
            x1, y1, x2, y2 = merged_box
            crop = img[y1:y2, x1:x2]

            # Run classification only for figures
            class_result = model_classify(crop, save=False)[0]
            pred_cls = int(class_result.probs.top1)
            pred_conf = float(class_result.probs.top1conf)

            if pred_cls == 0 and pred_conf >= 0.5:  # class 0 = chart
                chart_crops.append((crop, fconf, merged_box))

        # Remove duplicates after merging
        chart_crops = filter_duplicates(chart_crops)

        # Save unique charts
        for idx, (crop, _, box) in enumerate(chart_crops):
            out_path = os.path.join(output_charts, f"page{page_idx+1}_chart{idx}.png")
            cv2.imwrite(out_path, crop)
            print(f"✅ Saved chart: {out_path}")

        # ===============================
        # Handle Tables → save directly
        # ===============================
                # ===============================
        # Handle Tables → save directly
        # ===============================
        table_crops = []
        for tab_box, tconf in tables:
            x1, y1, x2, y2 = tab_box
            merged_box = tab_box.copy()

            # Merge with nearby caption
            for cap_box, _ in tab_captions:
                cx1, cy1, cx2, cy2 = cap_box
                if abs(cy1 - y2) < 150 or abs(y1 - cy2) < 150:
                    merged_box = [
                        min(merged_box[0], cx1),
                        min(merged_box[1], cy1),
                        max(merged_box[2], cx2),
                        max(merged_box[3], cy2),
                    ]

            # Expand crop region by 10% (left, right, top only)
            x1, y1, x2, y2 = merged_box
            w = x2 - x1
            h = y2 - y1
            pad_w = int(0.1 * w)
            pad_h = int(0.1 * h)

            x1 = max(0, x1 - pad_w)
            x2 = min(img.shape[1], x2 + pad_w)
            y1 = max(0, y1 - pad_h)
            # y2 remains same (no bottom padding)

            crop = img[y1:y2, x1:x2]
            table_crops.append((crop, tconf, [x1, y1, x2, y2]))

        # Remove duplicates after merging
        table_crops = filter_duplicates(table_crops)

        # Save unique tables
        for idx, (crop, _, box) in enumerate(table_crops):
            out_path = os.path.join(output_tables, f"page{page_idx+1}_table{idx}.png")
            cv2.imwrite(out_path, crop)
            print(f"✅ Saved table: {out_path}")

print("🎯 Finished! Charts and Tables saved separately (captions merged, duplicates/subsets removed).")

