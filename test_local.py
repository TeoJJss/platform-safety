from ultralytics import YOLO
import cv2
import numpy as np
import random, json, os, glob
import tempfile
from rfdetr import RFDETRNano

# ------------------------ CONFIG ------------------------
yolo_model_path = r"runs\platform-seg-yolo11.pt"
input_dir = r"input_images"
output_dir = r"output_images"
img_exts = (".jpg", ".jpeg", ".png")

os.makedirs(output_dir, exist_ok=True)
# YOLO-format output (images + labels)
# yolo_out_images = os.path.join(output_dir, 'yolo', 'images')
# yolo_out_labels = os.path.join(output_dir, 'yolo', 'labels')
# os.makedirs(yolo_out_images, exist_ok=True)
# os.makedirs(yolo_out_labels, exist_ok=True)

# ------------------------ LOAD ZONE SEGMENTATION MODEL ------------------------
model = YOLO(yolo_model_path)
print('Loading model names:', model.names)

# ------------------------ INIT PERSON DETECTION MODEL ------------------------
rf_model = None
try:
    rf_model_path = r"runs\crowd_detection_rf.pt"
    print('Initializing RFDETRNano model from:', rf_model_path)
    rf_model = RFDETRNano(pretrained_weights=rf_model_path)
    print('RFDETRNano model initialized.')
except Exception as e:
    rf_model = None
    print('Local RFDETRNano init failed or not available:', e)

FIXED_COLORS = {
    'danger zone': (50, 20, 50),   # Red
    'yellow line': (0, 20, 50), # Orange/Yellow
    'safe zone':   (50, 50, 0)  # Cyan
}
def process_image(img_path):
    print(f"Processing: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return
    H_img, W_img = img.shape[:2]

    results = model.predict(img_path, imgsz=800, conf=0.5, verbose=False, retina_masks=True)
    result = results[0]

    annotated_img = img.copy()
    # collect YOLO-format entries as tuples (class_id, x_center_norm, y_center_norm, w_norm, h_norm, polygon_coords[])
    # polygon_coords is a flat list [x1,y1,x2,y2,...] normalized to image size; empty list if none
    yolo_boxes = []
    masks_obj = getattr(result, 'masks', None)
    names = getattr(result, 'names', None) or getattr(model, 'names', None)

    # masks for overlap checking
    danger_mask = np.zeros((H_img, W_img), dtype=bool)
    yellow_mask = np.zeros((H_img, W_img), dtype=bool)
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
            try:
                cls_id = int(result.boxes.cls[i])
            except Exception:
                cls_id = 0
            cls_name = names[cls_id].lower() if names and cls_id in names else ''
            mask_i = masks[i]
            if mask_i.shape != (H_img, W_img):
                mask_i = cv2.resize((mask_i.astype('uint8') * 255), (W_img, H_img),
                                    interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask_i = mask_i > 0

            if cls_name in danger_classes:
                danger_mask = np.logical_or(danger_mask, mask_i)
            elif cls_name in yellow_classes:
                yellow_mask = np.logical_or(yellow_mask, mask_i)

            # if cls_id not in class_colors:
            #     class_colors[cls_id] = tuple(int(x) for x in rng.sample(range(50, 230), 3))
            color = FIXED_COLORS.get(cls_name, (255, 255, 255))

            overlay = annotated_img.copy()
            overlay[mask_i] = color
            annotated_img = cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0)

            ys, xs = np.where(mask_i)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
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
                # compute bounding box from mask polygon and add to YOLO boxes
                try:
                    x_min = int(xs.min())
                    x_max = int(xs.max())
                    y_min = int(ys.min())
                    y_max = int(ys.max())
                    bw = max(1, x_max - x_min)
                    bh = max(1, y_max - y_min)
                    cx_box = (x_min + x_max) / 2.0
                    cy_box = (y_min + y_max) / 2.0
                    # extract polygon contour points from mask
                    try:
                        mask_uint8 = (mask_i.astype('uint8') * 255)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # pick largest contour
                            contour = max(contours, key=cv2.contourArea)
                            # simplify contour to reduce point count
                            peri = cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
                            pts = approx.reshape(-1, 2)
                            poly_coords = []
                            for (px, py) in pts:
                                # normalize
                                poly_coords.append(float(px) / float(W_img))
                                poly_coords.append(float(py) / float(H_img))
                        else:
                            poly_coords = []
                    except Exception:
                        poly_coords = []

                    yolo_boxes.append((cls_id, cx_box / W_img, cy_box / H_img, bw / W_img, bh / H_img, poly_coords))
                except Exception:
                    pass

    # Roboflow predictions
    persons_in_danger = 0
    if rf_model is not None:
        try:
            # Some images have 4 channels (RGBA). RFDETR expects 3-channel images.
            # If source image has an alpha channel, create a temporary 3-channel copy and pass that path.
            tmp_img_path = None
            try:
                im_check = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if im_check is not None and im_check.ndim == 3 and im_check.shape[2] == 4:
                    tf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                    tf_name = tf.name
                    tf.close()
                    bgr = im_check[:, :, :3]
                    cv2.imwrite(tf_name, bgr)
                    tmp_img_path = tf_name
            except Exception:
                tmp_img_path = None

            use_path = tmp_img_path or img_path
            rf_resp = rf_model.predict(use_path, confidence=40, overlap=30)
            if tmp_img_path:
                try:
                    os.remove(tmp_img_path)
                except Exception:
                    pass

            # Normalize response into list of prediction dicts
            preds = []
            if isinstance(rf_resp, dict):
                preds = rf_resp.get('predictions', [])
            else:
                xyxy = getattr(rf_resp, 'xyxy', None)
                confs = getattr(rf_resp, 'confidence', None)
                class_ids = getattr(rf_resp, 'class_id', None)
                if xyxy is not None:
                    xy_arr = np.array(xyxy)
                    for i, box in enumerate(xy_arr):
                        try:
                            x1, y1, x2, y2 = [float(v) for v in box]
                        except Exception:
                            continue
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        w = x2 - x1
                        h = y2 - y1
                        label = None
                        if class_ids is not None:
                            try:
                                label = class_ids[i]
                            except Exception:
                                label = class_ids
                        conf = None
                        if confs is not None:
                            try:
                                conf = float(confs[i])
                            except Exception:
                                conf = None
                        preds.append({'x': cx, 'y': cy, 'width': w, 'height': h, 'class': label, 'confidence': conf})

            # filter preds to person class only (numeric id 1 or name containing 'person')
            person_preds = []
            for p in preds:
                cls = p.get('class', p.get('label', None))
                if cls is None:
                    continue
                matched = False
                try:
                    if int(cls) == 1:
                        matched = True
                except Exception:
                    pass
                try:
                    if 'person' in str(cls).lower():
                        matched = True
                except Exception:
                    pass
                if matched:
                    person_preds.append(p)

            print(f"Detections total: {len(preds)}, persons: {len(person_preds)}")
            preds = person_preds

            if preds:
                for p in preds:
                    x = p.get('x'); y = p.get('y'); w = p.get('width'); h = p.get('height')
                    label = p.get('class', p.get('label', 'obj'))
                    conf = p.get('confidence', None)
                    if None in (x, y, w, h):
                        continue

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

                    x1, x2 = max(0, x1), min(W_img - 1, x2)
                    y1, y2 = max(0, y1), min(H_img - 1, y2)

                    # Only consider the bottom 20% of the person's bbox for danger/yellow overlap
                    h_box = max(0, y2 - y1)
                    bottom_h = max(1, int(0.2 * h_box)) if h_box > 0 else 0
                    y_start = max(y1, y2 - bottom_h) if bottom_h > 0 else y1
                    # ensure valid slice ranges
                    if x1 >= x2 or y_start >= y2:
                        box_mask_danger = np.zeros((0,), dtype=bool)
                        box_mask_yellow = np.zeros((0,), dtype=bool)
                    else:
                        box_mask_danger = danger_mask[y_start:y2, x1:x2]
                        box_mask_yellow = yellow_mask[y_start:y2, x1:x2]

                    if np.any(box_mask_danger):
                        color = (0, 0, 255)
                        if str(label).lower() == 'person' or str(label) == '1':
                            persons_in_danger += 1
                    elif np.any(box_mask_yellow):
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

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
                    # do not add RF detections to YOLO labels — labels come from YOLO model only
        except Exception as e:
            print('Local RF inference failed:', e)

    out_name = os.path.basename(img_path)
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, annotated_img)
    print(f"Saved annotated image: {out_path} | persons_in_danger: {persons_in_danger}")

    # Also save a copy of the original image (not annotated) into YOLO images folder
    # try:
    #     img_out_path = os.path.join(yolo_out_images, out_name)
    #     cv2.imwrite(img_out_path, img)
    # except Exception as e:
    #     print('Failed to save YOLO image copy:', e)

    # If RF model wasn't used or found no persons, try to extract person boxes from YOLO result boxes
    if not yolo_boxes:
        try:
            boxes = getattr(result, 'boxes', None)
            names = getattr(result, 'names', None) or getattr(model, 'names', None)
            if boxes is not None:
                xyxy = getattr(boxes, 'xyxy', None)
                cls_arr = getattr(boxes, 'cls', None)
                if xyxy is not None:
                    xy = np.array(xyxy)
                    for i, b in enumerate(xy):
                        try:
                            x1, y1, x2, y2 = [float(v) for v in b]
                        except Exception:
                            continue
                        cls_id = 0
                        try:
                            if cls_arr is not None:
                                cls_id = int(cls_arr[i])
                        except Exception:
                            pass
                        cls_name = ''
                        try:
                            cls_name = names[cls_id].lower()
                        except Exception:
                            pass
                        cw = (x1 + x2) / 2.0
                        ch = (y1 + y2) / 2.0
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)
                        # fallback: append without polygon (YOLO boxes only)
                        yolo_boxes.append((cls_id, cw / W_img, ch / H_img, bw / W_img, bh / H_img, []))
        except Exception:
            pass

    # Write YOLO label file (same basename, .txt) with one line per detection
    # try:
    #     label_name = os.path.splitext(out_name)[0] + '.txt'
    #     # label_path = os.path.join(yolo_out_labels, label_name)
    #     with open(label_path, 'w', encoding='utf-8') as f:
    #         for entry in yolo_boxes:
    #             # entry: (cls_id, xc, yc, w, h, poly_coords)
    #             try:
    #                 cls_id, xc, yc, w_n, h_n, poly = entry
    #             except Exception:
    #                 # backwards compatibility: accept 5-tuples
    #                 try:
    #                     cls_id, xc, yc, w_n, h_n = entry
    #                     poly = []
    #                 except Exception:
    #                     continue
    #             line = f"{int(cls_id)} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}"
    #             if poly:
    #                 # append polygon coordinates
    #                 line += " " + " ".join(f"{v:.6f}" for v in poly)
    #             f.write(line + "\n")
    # except Exception as e:
    #     print('Failed to write YOLO labels:', e)


def main():
    files = []
    for ext in img_exts:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    files = sorted(files)
    if not files:
        print(f"No images found in {input_dir}")
        return
    for idx, fp in enumerate(files, 1):
        try:
            process_image(fp)
        except Exception as e:
            print(f"Failed processing {fp}: {e}")


if __name__ == '__main__':
    main()
