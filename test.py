from ultralytics import YOLO
import cv2
import numpy as np
import random, json
from roboflow import Roboflow

# ------------------------ CONFIG ------------------------
yolo_model_path = r"runs\segment\train3\weights\best.pt"
img_path = r"awanbesar1.jpeg"
output_path = "labeled_test.jpg"

# ------------------------ LOAD YOLO ------------------------
model = YOLO(yolo_model_path)
print('Loading model names:', model.names)

# ------------------------ LOAD IMAGE ------------------------
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")
H_img, W_img = img.shape[:2]

# ------------------------ PREDICT YOLO ------------------------
results = model.predict(img_path, imgsz=640, conf=0.1, verbose=False)
result = results[0]

# ------------------------ CREATE ANNOTATED IMAGE ------------------------
annotated_img = img.copy()
masks_obj = getattr(result, 'masks', None)
names = getattr(result, 'names', None) or getattr(model, 'names', None)

# ------------------------ PROCESS MASKS ------------------------
danger_mask = np.zeros((H_img, W_img), dtype=bool)  # for overlap checking (danger zone)
yellow_mask = np.zeros((H_img, W_img), dtype=bool)   # for yellow line checking
danger_classes = ['danger zone']
yellow_classes = ['yellow line']

masks = None
if masks_obj is not None:
    if hasattr(masks_obj, 'data') and masks_obj.data is not None:
        masks = masks_obj.data.cpu().numpy()
    elif hasattr(masks_obj, 'masks') and masks_obj.masks is not None:
        masks = masks_obj.masks.cpu().numpy()

if masks is not None:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, ...]
    if masks.ndim == 3 and masks.shape[0] == img.shape[0] and masks.shape[1] == img.shape[1]:
        masks = masks.transpose(2, 0, 1)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    rng = random.Random(0)
    class_colors = {}

    for i in range(masks.shape[0]):
        cls_id = int(result.boxes.cls[i])
        cls_name = names[cls_id].lower() if names and cls_id in names else ''
        mask_i = masks[i]
        if mask_i.shape != (H_img, W_img):
            mask_i = cv2.resize((mask_i.astype('uint8') * 255), (W_img, H_img),
                                interpolation=cv2.INTER_NEAREST) > 0
        else:
            mask_i = mask_i > 0

        # Add masks to their respective masks
        if cls_name in danger_classes:
            danger_mask = np.logical_or(danger_mask, mask_i)
        elif cls_name in yellow_classes:
            yellow_mask = np.logical_or(yellow_mask, mask_i)

        # assign color for this class if not exists
        if cls_id not in class_colors:
            class_colors[cls_id] = tuple(int(x) for x in rng.sample(range(50, 230), 3))
        color = class_colors[cls_id]

        # Draw mask overlay
        overlay = annotated_img.copy()
        overlay[mask_i] = color
        annotated_img = cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0)

        # Draw bounding box if exists
        # try:
        #     xyxy = result.boxes.xyxy[i]
        #     x1, y1, x2, y2 = [int(x) for x in xyxy]
        #     cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        # except Exception:
        #     x1, y1 = 0, 0  # fallback for label positioning

        # Draw label text + confidence at centroid of mask (skip text for 'head')
        ys, xs = np.where(mask_i)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            # only draw text if the class is not 'head'
            if cls_name.lower() != 'head':
                label_text = cls_name.title()
                try:
                    conf = float(result.boxes.conf[i])
                    label_text += f" {conf:.2f}"
                except Exception:
                    pass
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                pad = 4
                bx1 = max(0, cx - text_w // 2 - pad)
                by1 = max(0, cy - text_h - baseline - pad)
                bx2 = min(W_img - 1, bx1 + text_w + pad*2)
                by2 = min(H_img - 1, by1 + text_h + baseline + pad*2)
                cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0,0,0), -1)
                cv2.putText(annotated_img, label_text, (bx1 + pad, by2 - baseline - pad),
                            font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

# ------------------------ INIT ROBOFLOW ------------------------
try:
    secret = json.load(open('secret.json'))
    rf = Roboflow(api_key=secret["roboflow"]["api_key"])
    rf_project = rf.workspace().project(secret["roboflow"]["project_name"])
    rf_model = rf_project.version("1").model
except Exception as e:
    rf_model = None
    print('Roboflow init failed or not available:', e)

# ------------------------ ROBOFLOW PREDICTIONS ------------------------
persons_in_danger = 0
if rf_model is not None:
    try:
        rf_resp = rf_model.predict(img_path, confidence=40, overlap=30).json()
        preds = rf_resp.get('predictions', []) if isinstance(rf_resp, dict) else []
        if preds:
            for p in preds:
                x = p.get('x'); y = p.get('y'); w = p.get('width'); h = p.get('height')
                label = p.get('class', p.get('label', 'obj'))
                conf = p.get('confidence', None)
                if None in (x, y, w, h):
                    continue

                # Convert to pixels
                if 0.0 < x <= 1.0:
                    x1 = int((x - w / 2) * W_img)
                    y1 = int((y - h / 2) * H_img)
                    x2 = int((x + w / 2) * W_img)
                    y2 = int((y + h / 2) * H_img)
                else:
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                # Clamp coordinates
                x1, x2 = max(0, x1), min(W_img - 1, x2)
                y1, y2 = max(0, y1), min(H_img - 1, y2)

                # Check overlap with danger and yellow masks
                box_mask_danger = danger_mask[y1:y2, x1:x2]
                box_mask_yellow = yellow_mask[y1:y2, x1:x2]
                if np.any(box_mask_danger):
                    color = (0, 0, 255)  # red if danger zone (BGR)
                    if str(label).lower() == 'person':
                        persons_in_danger += 1
                elif np.any(box_mask_yellow):
                    color = (0, 165, 255)  # orange if on yellow line (BGR)
                else:
                    color = (0, 255, 0)  # green if safe

                # Draw box and label, but skip both if class is 'head'
                if str(label).lower() != 'head':
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label} {conf:.2f}" if conf is not None else str(label)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    pad = 4
                    bx1 = max(0, x1)
                    by1 = max(0, y1 - text_h - baseline - pad)
                    bx2 = min(W_img - 1, bx1 + text_w + pad*2)
                    by2 = min(H_img - 1, by1 + text_h + baseline + pad*2)
                    cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                    cv2.putText(annotated_img, label_text, (bx1 + pad, by2 - baseline - pad),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                else:
                    # intentionally skip drawing bounding box and label for 'head'
                    pass
            # print('Roboflow predictions drawn.')
        else:
            print('Roboflow returned no predictions.')
    except Exception as e:
        print('Roboflow inference failed:', e)

# ------------------------ SAVE IMAGE ------------------------
cv2.imwrite(output_path, annotated_img)
print(f'Saved annotated image: {output_path}')
print(f"Number of persons in danger zone: {persons_in_danger}")
