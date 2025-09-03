#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Load models using relative paths
model_detect = YOLOv10("models/doclayout_yolo_docstructbench_imgsz1024.pt")  # detection model
model_classify = YOLO("models/classification_chart.pt")  # classification model


# In[25]:


# ===============================
# Config
# ===============================
input_pdf = "input_pdfs/Edgewater - Memory Insights (December-22).pdf"

output_charts = "output_charts"
output_tables = "output_tables"
os.makedirs(output_charts, exist_ok=True)
os.makedirs(output_tables, exist_ok=True)


# In[26]:


# Detection config (adjust indices to your model)
FIGURE_CLASS = 3
FIGURE_CAPTION_CLASS = 4
TABLE_CLASS = 5
TABLE_CAPTION_CLASS = 6
CONF_THRESHOLD = 0.1

# ===============================
# PDF → Images
# ===============================
pages = convert_from_path(input_pdf)

for page_idx, page in enumerate(pages):
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

    # 🔹 Run detection once
    results = model_detect(img, save=False)

    for r in results:
        boxes = r.boxes

        # Separate detections
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
        for i, (fig_box, fconf) in enumerate(figures):
            x1, y1, x2, y2 = fig_box
            merged_box = fig_box.copy()

            for cap_box, cconf in fig_captions:
                cx1, cy1, cx2, cy2 = cap_box
                if abs(cy1 - y2) < 150 or abs(y1 - cy2) < 150:
                    merged_box = [
                        min(merged_box[0], cx1),
                        min(merged_box[1], cy1),
                        max(merged_box[2], cx2),
                        max(merged_box[3], cy2),
                    ]

            x1, y1, x2, y2 = merged_box
            crop = img[y1:y2, x1:x2]

            # 🔹 Run classification only for figures
            class_result = model_classify(crop, save=False)[0]
            pred_cls = int(class_result.probs.top1)
            pred_conf = float(class_result.probs.top1conf)

            # Assuming class 0 = "chart"
            if pred_cls == 0 and pred_conf >= 0.5:
                out_path = os.path.join(output_charts, f"page{page_idx+1}_figure{i}_chart.png")
                cv2.imwrite(out_path, crop)
                print(f"✅ Saved chart: {out_path}")

        # ===============================
        # Handle Tables → save directly
        # ===============================
        for j, (tab_box, tconf) in enumerate(tables):
            x1, y1, x2, y2 = tab_box
            merged_box = tab_box.copy()

            for cap_box, cconf in tab_captions:
                cx1, cy1, cx2, cy2 = cap_box
                if abs(cy1 - y2) < 150 or abs(y1 - cy2) < 150:
                    merged_box = [
                        min(merged_box[0], cx1),
                        min(merged_box[1], cy1),
                        max(merged_box[2], cx2),
                        max(merged_box[3], cy2),
                    ]

            x1, y1, x2, y2 = merged_box
            crop = img[y1:y2, x1:x2]

            out_path = os.path.join(output_tables, f"page{page_idx+1}_table{j}.png")
            cv2.imwrite(out_path, crop)
            print(f"✅ Saved table: {out_path}")

print("🎯 Finished! Charts and Tables saved separately.")

