import sys
import os
import cv2
import numpy as np
from rfdetr import RFDETRNano

# Simple script to run the RFDETRNano model and label one image.
# Usage: python label_person.py <input_image> [output_image]

MODEL_PATH = r"runs\crowd_detection_rf.pt"


def rf_resp_to_preds(rf_resp):
    """Normalize rf_resp (dict or Detections-like) into list of preds dicts.
    Each pred dict: {'x','y','width','height','class','confidence'}
    x,y can be either normalized center (0..1) or absolute pixels depending on source.
    """
    preds = []
    if isinstance(rf_resp, dict):
        preds = rf_resp.get('predictions', []) or []
    else:
        xyxy = getattr(rf_resp, 'xyxy', None)
        confs = getattr(rf_resp, 'confidence', None)
        class_ids = getattr(rf_resp, 'class_id', None)

        if xyxy is not None:
            arr = np.array(xyxy)
            for i, box in enumerate(arr):
                try:
                    x1, y1, x2, y2 = [float(v) for v in box]
                except Exception:
                    continue
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                lbl = None
                if class_ids is not None:
                    try:
                        lbl = class_ids[i]
                    except Exception:
                        lbl = class_ids

                try:
                    conf_val = float(confs[i]) if confs is not None else None
                except Exception:
                    conf_val = None

                preds.append({
                    'x': cx,
                    'y': cy,
                    'width': w,
                    'height': h,
                    'class': str(lbl) if lbl is not None else 'obj',
                    'confidence': conf_val,
                })

    return preds


def preds_to_xyxy_pixels(preds, W, H):
    """Convert preds entries to pixel xyxy boxes.
    Accepts preds where x,y may be normalized (0..1) or absolute pixels.
    Returns list of tuples (x1,y1,x2,y2,label,confidence)
    """
    boxes = []
    for p in preds:
        x = p.get('x'); y = p.get('y'); w = p.get('width'); h = p.get('height')
        if None in (x, y, w, h):
            continue

        try:
            xf = float(x); yf = float(y); wf = float(w); hf = float(h)
        except Exception:
            continue

        # If center coords look normalized use W/H to convert
        if 0.0 < xf <= 1.0 and 0.0 < yf <= 1.0 and 0.0 < wf <= 1.0 and 0.0 < hf <= 1.0:
            cx = xf * W
            cy = yf * H
            width = wf * W
            height = hf * H
        else:
            cx = xf
            cy = yf
            width = wf
            height = hf

        x1 = int(round(cx - width / 2.0))
        y1 = int(round(cy - height / 2.0))
        x2 = int(round(cx + width / 2.0))
        y2 = int(round(cy + height / 2.0))

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W - 1, x2); y2 = min(H - 1, y2)

        boxes.append((x1, y1, x2, y2, p.get('class'), p.get('confidence')))

    return boxes


def draw_boxes(img, boxes):
    out = img.copy()
    for (x1, y1, x2, y2, label, conf) in boxes:
        # color: red for person keyword, else green
        lab = str(label).lower() if label is not None else ''
        if 'person' in lab or lab == '1':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_text = str(label)
        if conf is not None:
            try:
                label_text = f"{label_text} {float(conf):.2f}"
            except Exception:
                pass

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        pad = 4
        bx1 = max(0, x1)
        by1 = max(0, y1 - text_h - baseline - pad)
        bx2 = min(out.shape[1] - 1, bx1 + text_w + pad * 2)
        by2 = min(out.shape[0] - 1, by1 + text_h + baseline + pad * 2)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.putText(out, label_text, (bx1 + pad, by2 - baseline - pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def is_person_pred(p, person_ids=(1,), person_names=('person',)):
    """Return True if pred `p` corresponds to a person.
    Supports numeric class ids (e.g. 1) and class name strings containing 'person'.
    """
    cls = p.get('class')
    if cls is None:
        return False
    # numeric match
    try:
        if int(cls) in person_ids:
            return True
    except Exception:
        pass
    # name match
    try:
        s = str(cls).lower()
        for name in person_names:
            if name in s:
                return True
    except Exception:
        pass
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python label_person.py <input_image> [output_image]")
        return

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}")
        return

    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(input_path), 'labeled_' + os.path.basename(input_path))

    try:
        print(f"Loading RF model from: {MODEL_PATH}")
        rf_model = RFDETRNano(pretrained_weights=MODEL_PATH)
    except Exception as e:
        print('Failed to initialize RFDETRNano model:', e)
        return

    try:
        print('Running inference...')
        rf_resp = rf_model.predict(input_path, confidence=40, overlap=30)
        print('RF response:', type(rf_resp))
    except Exception as e:
        print('Inference failed:', e)
        return

    img = cv2.imread(input_path)
    if img is None:
        print('Failed to read input image')
        return
    H, W = img.shape[:2]

    preds = rf_resp_to_preds(rf_resp)
    # filter to person class only
    person_preds = [p for p in preds if is_person_pred(p)]
    print(f"Detections total: {len(preds)}, persons: {len(person_preds)}")
    boxes = preds_to_xyxy_pixels(person_preds, W, H)

    if not boxes:
        print('No detections found, writing original image.')
        cv2.imwrite(out_path, img)
        print('Saved:', out_path)
        return

    annotated = draw_boxes(img, boxes)
    cv2.imwrite(out_path, annotated)
    print('Saved annotated image to:', out_path)


if __name__ == '__main__':
    main()
