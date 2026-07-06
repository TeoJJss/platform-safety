import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import winsound
except Exception:
    winsound = None


class PlatformSafetyEngine:
    """Engine that runs zone segmentation and person detection.
    """
    def __init__(self, yolo_model_path, rf_model_path, seg_model=None, crowd_model=None):
        self.yolo_model_path = yolo_model_path
        self.rf_model_path = rf_model_path

        self.fixed_colors = {
            "danger zone": (50, 20, 50),      # Dark red
            "yellow line": (0, 20, 50),       # Dark orange
            "safe zone": (50, 50, 0),         # Dark green
        }
        self.danger_classes = {"danger zone"}
        self.yellow_classes = {"yellow line"}

        if seg_model is None:
            from ultralytics import YOLO

            seg_model = YOLO(self.yolo_model_path)

        if crowd_model is None:
            from rfdetr import RFDETRNano

            crowd_model = RFDETRNano(pretrained_weights=self.rf_model_path)

        self.seg_model = seg_model
        self.crowd_model = crowd_model

    @staticmethod
    def _normalize_masks(masks_obj, width, height):
        """Normalize different mask object formats into a 3D boolean array.
        """
        if masks_obj is None:
            return None

        masks = None
        if hasattr(masks_obj, "data") and masks_obj.data is not None:
            masks = masks_obj.data.cpu().numpy()
        elif hasattr(masks_obj, "masks") and masks_obj.masks is not None:
            masks = masks_obj.masks.cpu().numpy()

        if masks is None:
            return None

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, ...]
        if masks.ndim == 3 and masks.shape[0] == height and masks.shape[1] == width:
            masks = masks.transpose(2, 0, 1)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]

        return masks

    @staticmethod
    def _parse_rf_predictions(rf_resp):
        """Parse RF-DETR predictions into a list of simple dicts.

        Each dict has keys: x,y,width,height,class,confidence
        """
        preds = []

        if isinstance(rf_resp, dict):
            return rf_resp.get("predictions", [])

        xyxy = getattr(rf_resp, "xyxy", None)
        confs = getattr(rf_resp, "confidence", None)
        class_ids = getattr(rf_resp, "class_id", None)

        if xyxy is None:
            return preds

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

            cls_val = None
            if class_ids is not None:
                try:
                    cls_val = class_ids[i]
                except Exception:
                    cls_val = class_ids

            conf = None
            if confs is not None:
                try:
                    conf = float(confs[i])
                except Exception:
                    conf = None

            preds.append(
                {
                    "x": cx,
                    "y": cy,
                    "width": w,
                    "height": h,
                    "class": cls_val,
                    "confidence": conf,
                }
            )

        return preds

    @staticmethod
    def _is_person_prediction(pred):
        """
        Return True if prediction represents a person.
        """
        cls = pred.get("class", pred.get("label", None))
        if cls is None:
            return False

        try:
            if int(cls) == 1:
                return True
        except Exception:
            pass

        try:
            if "person" in str(cls).lower():
                return True
        except Exception:
            pass

        return False

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union

    def _run_seg_inference(self, frame, imgsz=800):
        """Run YOLO segmentation inference in a thread."""
        seg_start = time.perf_counter()
        result = self.seg_model.predict(
            frame,
            imgsz=imgsz,
            conf=0.5,
            verbose=False,
            retina_masks=True,
        )[0]
        seg_infer_ms = (time.perf_counter() - seg_start) * 1000.0
        return result, seg_infer_ms

    def _run_rf_inference(self, frame):
        """Run RF-DETR person detection inference in a thread."""
        rf_start = time.perf_counter()
        rf_resp = self.crowd_model.predict(frame, confidence=40, overlap=30)
        rf_infer_ms = (time.perf_counter() - rf_start) * 1000.0
        return rf_resp, rf_infer_ms

    def process_frame(self, frame, show_zone_masks=True, imgsz=800):
        # record start time for profiling
        start = time.perf_counter()
        h_img, w_img = frame.shape[:2]
        annotated_img = frame.copy()

        danger_mask = np.zeros((h_img, w_img), dtype=bool)
        yellow_mask = np.zeros((h_img, w_img), dtype=bool)

        # Run YOLO and RF-DETR inferences in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            seg_future = executor.submit(self._run_seg_inference, frame, imgsz)
            rf_future = executor.submit(self._run_rf_inference, frame)
            
            # Wait for both to complete
            result, seg_infer_ms = seg_future.result()
            rf_resp, rf_infer_ms = rf_future.result()
        
        print(f"[YOLO ZONE SEGMENTATION] Inference time: {seg_infer_ms:.2f} ms")
        print(f"[RFDETR PERSON DETECTION] Inference time: {rf_infer_ms:.2f} ms")

        names = getattr(result, "names", None) or getattr(self.seg_model, "names", None)
        masks = self._normalize_masks(getattr(result, "masks", None), w_img, h_img)
        # collect confidence scores for detected zone masks
        zone_confidence_values = []

        # if segmentation masks present, process each mask
        if masks is not None:
            for i in range(masks.shape[0]):
                try:
                    cls_id = int(result.boxes.cls[i])
                except Exception:
                    cls_id = 0

                conf_score = None
                try:
                    conf_score = float(result.boxes.conf[i])
                except Exception:
                    conf_score = None
                
                if conf_score is not None:
                    zone_confidence_values.append(conf_score)

                cls_name = ""
                try:
                    cls_name = names[cls_id].lower()
                except Exception:
                    cls_name = ""

                mask_i = masks[i]
                if mask_i.shape != (h_img, w_img):
                    mask_i = cv2.resize(
                        (mask_i.astype("uint8") * 255),
                        (w_img, h_img),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                else:
                    mask_i = mask_i > 0

                if cls_name in self.danger_classes:
                    danger_mask = np.logical_or(danger_mask, mask_i)
                elif cls_name in self.yellow_classes:
                    yellow_mask = np.logical_or(yellow_mask, mask_i)

                # optionally draw mask overlay and label on the image
                if show_zone_masks:
                    color = self.fixed_colors.get(cls_name, (255, 255, 255))
                    overlay = annotated_img.copy()
                    overlay[mask_i] = color
                    annotated_img = cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0)

                    # Place a readable label near the mask centroid if mask exists
                    if np.any(mask_i):
                        ys, xs = np.where(mask_i)
                        cx = int(np.mean(xs))
                        cy = int(np.mean(ys))
                        base_label = cls_name if cls_name else f"class {cls_id}"
                        label_text = f"{base_label} {conf_score:.2f}" if conf_score is not None else base_label

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.55
                        thickness = 2
                        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                        tx = max(3, min(w_img - tw - 3, cx - tw // 2))
                        ty = max(th + 6, min(h_img - baseline - 3, cy))

                        cv2.rectangle(
                            annotated_img,
                            (tx - 3, ty - th - 4),
                            (tx + tw + 3, ty + baseline + 2),
                            color,
                            -1,
                        )
                        text_color = (0, 0, 0) if cls_name == "train" else (255, 255, 255)
                        cv2.putText(
                            annotated_img,
                            label_text,
                            (tx, ty),
                            font,
                            font_scale,
                            text_color,
                            thickness,
                            cv2.LINE_AA,
                        )
        preds = self._parse_rf_predictions(rf_resp)
        person_preds = [p for p in preds if self._is_person_prediction(p)]

        people_detected = len(person_preds)
        people_in_danger = 0
        people_on_warning = 0
        person_boxes = []
        confidence_values = []

        for p in person_preds:
            x = p.get("x")
            y = p.get("y")
            w = p.get("width")
            h = p.get("height")
            label = p.get("class", p.get("label", "person"))
            conf = p.get("confidence", None)
            if None in (x, y, w, h):
                continue

            if 0.0 < x <= 1.0:
                x1 = int((x - w / 2) * w_img)
                y1 = int((y - h / 2) * h_img)
                x2 = int((x + w / 2) * w_img)
                y2 = int((y + h / 2) * h_img)
            else:
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

            x1, x2 = max(0, x1), min(w_img - 1, x2)
            y1, y2 = max(0, y1), min(h_img - 1, y2)

            h_box = max(0, y2 - y1)
            bottom_h = max(1, int(0.2 * h_box)) if h_box > 0 else 0
            y_start = max(y1, y2 - bottom_h) if bottom_h > 0 else y1

            if x1 >= x2 or y_start >= y2:
                box_mask_danger = np.zeros((0,), dtype=bool)
                box_mask_yellow = np.zeros((0,), dtype=bool)
            else:
                box_mask_danger = danger_mask[y_start:y2, x1:x2]
                box_mask_yellow = yellow_mask[y_start:y2, x1:x2]

            if np.any(box_mask_danger):
                color = (0, 0, 255)
                people_in_danger += 1
            elif np.any(box_mask_yellow):
                color = (0, 165, 255)
                people_on_warning += 1
            else:
                color = (0, 255, 0)

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            label_text = f"Person {conf:.2f}" if conf is not None else str(label)
            cv2.putText(
                annotated_img,
                label_text,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            person_boxes.append((x1, y1, x2, y2))
            if conf is not None:
                confidence_values.append(float(conf))

        people_safe = max(0, people_detected - people_in_danger - people_on_warning)

        total_pixels = float(h_img * w_img)
        danger_coverage_pct = float(np.count_nonzero(danger_mask)) * 100.0 / total_pixels
        yellow_coverage_pct = float(np.count_nonzero(yellow_mask)) * 100.0 / total_pixels
        zones_detected = danger_coverage_pct > 0.0 or yellow_coverage_pct > 0.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        metrics = {
            "people_detected": int(people_detected),
            "people_in_danger": int(people_in_danger),
            "people_on_warning": int(people_on_warning),
            "people_safe": int(people_safe),
            "risk_index": float(people_in_danger / people_detected) if people_detected else 0.0,
            "danger_zone_coverage_pct": danger_coverage_pct,
            "yellow_zone_coverage_pct": yellow_coverage_pct,
            "zones_detected": zones_detected,
            "response_time_ms": elapsed_ms,
            "avg_confidence": float(np.mean(confidence_values)) if confidence_values else None,
            "avg_zone_confidence": float(np.mean(zone_confidence_values)) if zone_confidence_values else None,
        }

        details = {
            "person_boxes": person_boxes,
        }
        return annotated_img, metrics, details


class PlatformSafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Railway Platform Safety Control Center")
        self.root.state("zoomed")
        self.root.configure(bg="#0E1726")

        self.engine = None
        self.current_input_path = None
        self.current_is_video = False
        self.current_original = None
        self.current_annotated = None
        self.current_metrics = {}
        self.output_path = None
        self.show_masks_var = tk.BooleanVar(value=True)
        self.panel_zoom = {"input": 1.0, "output": 1.0}
        self.panel_images = {"input": None, "output": None}
        self.zoom_min = 0.5
        self.zoom_max = 4.0
        self.last_alert_level = None
        self.last_alert_message = ""
        self.alert_counter = 0
        self.video_frames_since_last_alert = 999
        self.flash_after_id = None
        self.flash_on = False

        self.playback_cap = None
        self.playback_after_id = None
        self.playback_active = False

        self.auto_process_pending = False

        self.stop_event = threading.Event()
        self.ui_queue = Queue()
        self.worker_thread = None
        self.model_thread = None
        self.model_ready = False

        self.output_dir = Path("output_images")
        self.output_video_dir = Path("output_video")
        self.report_dir = Path("output_reports")

        self._build_ui()
        self._start_model_loading()
        self._poll_queue()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#122033")
        style.configure("Card.TFrame", background="#182A40")
        style.configure("Metric.TLabel", background="#182A40", foreground="#E6EDF6", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#122033", foreground="#F3F6FB", font=("Segoe UI Semibold", 12))
        style.configure("Value.TLabel", background="#182A40", foreground="#8BE9FD", font=("Consolas", 12, "bold"))

        root_frame = ttk.Frame(self.root, style="Panel.TFrame", padding=10)
        root_frame.pack(fill="both", expand=True)

        control_frame = ttk.Frame(root_frame, style="Card.TFrame", padding=10)
        control_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(control_frame, text="Platform Safety Operator Console", style="Header.TLabel").pack(side="left")

        ttk.Button(control_frame, text="Select Image / Video", command=self.select_input).pack(side="left", padx=10)
        ttk.Checkbutton(
            control_frame,
            text="Show Zone Masks",
            variable=self.show_masks_var,
        ).pack(side="left", padx=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Initializing models...")
        self.status_label = tk.Label(
            control_frame,
            textvariable=self.status_var,
            bg="#182A40",
            fg="#FDFDFD",
            padx=10,
            pady=6,
            relief="groove",
        )
        self.status_label.pack(side="right")

        self.severity_var = tk.StringVar(value="IDLE")
        self.severity_label = tk.Label(
            control_frame,
            textvariable=self.severity_var,
            bg="#243447",
            fg="#FFFFFF",
            padx=10,
            pady=6,
            relief="groove",
            font=("Segoe UI", 10, "bold"),
        )
        self.severity_label.pack(side="right", padx=(0, 8))

        self.alert_var = tk.StringVar(value="")
        self.alert_label = tk.Label(
            root_frame,
            textvariable=self.alert_var,
            bg="#0E1726",
            fg="#FFFFFF",
            font=("Segoe UI Semibold", 13, "bold"),
            padx=12,
            pady=8,
            relief="flat",
        )
        self.alert_label.pack(fill="x", pady=(0, 10))
        self.alert_label.pack_forget()

        self.alert_meta_var = tk.StringVar(value="")
        self.alert_meta_label = tk.Label(
            root_frame,
            textvariable=self.alert_meta_var,
            bg="#0E1726",
            fg="#D6DEE8",
            anchor="w",
            padx=12,
            pady=4,
        )
        self.alert_meta_label.pack(fill="x", pady=(0, 10))
        self.alert_meta_label.pack_forget()

        content = ttk.Frame(root_frame, style="Panel.TFrame")
        content.pack(fill="both", expand=True)

        view_frame = ttk.Frame(content, style="Card.TFrame", padding=8)
        view_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right_panel = ttk.Frame(content, style="Card.TFrame", padding=8)
        right_panel.pack(side="right", fill="y")

        metrics_canvas = tk.Canvas(right_panel, bg="#182A40", highlightthickness=0, width=320)
        metrics_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=metrics_canvas.yview)
        metrics_canvas.configure(yscrollcommand=metrics_scrollbar.set)

        metrics_scrollbar.pack(side="right", fill="y")
        metrics_canvas.pack(side="left", fill="both", expand=True)

        metrics_body = ttk.Frame(metrics_canvas, style="Card.TFrame")
        metrics_window = metrics_canvas.create_window((0, 0), window=metrics_body, anchor="nw")

        def sync_scrollregion(_event=None):
            metrics_canvas.configure(scrollregion=metrics_canvas.bbox("all"))
            metrics_canvas.itemconfigure(metrics_window, width=metrics_canvas.winfo_width())

        metrics_body.bind("<Configure>", sync_scrollregion)
        metrics_canvas.bind("<Configure>", sync_scrollregion)
        metrics_canvas.bind_all(
            "<MouseWheel>",
            lambda event: metrics_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"),
        )

        ttk.Label(view_frame, text="Input vs Labeled Output", style="Header.TLabel").pack(anchor="w", pady=(0, 8))

        images_row = ttk.Frame(view_frame, style="Card.TFrame")
        images_row.pack(fill="both", expand=True)

        self.input_panel = tk.Label(images_row, bg="#0A111A", fg="#D6DEE8", text="Input Preview", width=60, height=28)
        self.input_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.input_panel.bind("<MouseWheel>", lambda event: self._on_panel_zoom("input", event))
        self.input_panel.bind("<Configure>", lambda event: self._on_panel_resize("input", event))

        self.output_panel = tk.Label(images_row, bg="#0A111A", fg="#D6DEE8", text="Labeled Preview", width=60, height=28)
        self.output_panel.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.output_panel.bind("<MouseWheel>", lambda event: self._on_panel_zoom("output", event))
        self.output_panel.bind("<Configure>", lambda event: self._on_panel_resize("output", event))

        self.path_var = tk.StringVar(value="No input selected")
        ttk.Label(view_frame, textvariable=self.path_var, style="Metric.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(metrics_body, text="Operational Metrics", style="Header.TLabel").pack(anchor="w", pady=(0, 8))

        self.metric_vars = {
            "people_detected": tk.StringVar(value="0"),
            "people_in_danger": tk.StringVar(value="0"),
            "people_on_warning": tk.StringVar(value="0"),
            "people_safe": tk.StringVar(value="0"),
            "risk_index": tk.StringVar(value="0.00"),
            "danger_zone_coverage_pct": tk.StringVar(value="0.00%"),
            "yellow_zone_coverage_pct": tk.StringVar(value="0.00%"),
            "response_time_ms": tk.StringVar(value="0.00 ms"),
            "avg_confidence": tk.StringVar(value="N/A"),
            "avg_zone_confidence": tk.StringVar(value="N/A"),
        }

        metric_labels = [
            ("People Detected", "people_detected"),
            ("People In Danger", "people_in_danger"),
            ("People On Warning", "people_on_warning"),
            ("People Safe", "people_safe"),
            ("Risk Index", "risk_index"),
            ("Danger Zone Coverage", "danger_zone_coverage_pct"),
            ("Yellow Zone Coverage", "yellow_zone_coverage_pct"),
            ("Response Time", "response_time_ms"),
            ("Avg Confidence (Person)", "avg_confidence"),
            ("Avg Confidence (Zone)", "avg_zone_confidence"),
        ]

        for title, key in metric_labels:
            card = ttk.Frame(metrics_body, style="Card.TFrame", padding=6)
            card.pack(fill="x", pady=3)
            ttk.Label(card, text=title, style="Metric.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.metric_vars[key], style="Value.TLabel").pack(anchor="w")

        ttk.Label(metrics_body, text="Incident Log", style="Header.TLabel").pack(anchor="w", pady=(10, 6))
        self.log_box = tk.Text(
            metrics_body,
            height=14,
            width=44,
            bg="#0A111A",
            fg="#D6DEE8",
            insertbackground="#D6DEE8",
            relief="flat",
        )
        self.log_box.pack(fill="both", expand=True)

    def _start_model_loading(self):
        self.status_var.set("Loading models in parallel...")
        if self.model_thread and self.model_thread.is_alive():
            return
        self.model_thread = threading.Thread(target=self._load_models, daemon=True)
        self.model_thread.start()

    def _load_models(self):
        try:
            from rfdetr import RFDETRNano
            from ultralytics import YOLO

            yolo_path = "./runs/platform-seg-yolo11_nano.pt"
            rf_path = "./runs/crowd_rfdetr-nano.pt"
            model_results = {}
            errors = Queue()

            def load_yolo():
                try:
                    model_results["seg_model"] = YOLO(yolo_path)
                except Exception as exc:
                    errors.put(("YOLO", exc))

            def load_rf():
                try:
                    model_results["crowd_model"] = RFDETRNano(pretrained_weights=rf_path)
                except Exception as exc:
                    errors.put(("RFDETR", exc))

            yolo_thread = threading.Thread(target=load_yolo, daemon=True)
            rf_thread = threading.Thread(target=load_rf, daemon=True)
            yolo_thread.start()
            rf_thread.start()
            yolo_thread.join()
            rf_thread.join()

            if not errors.empty():
                model_name, error = errors.get()
                raise RuntimeError(f"{model_name} model failed to load: {error}")

            engine = PlatformSafetyEngine(
                yolo_path,
                rf_path,
                seg_model=model_results.get("seg_model"),
                crowd_model=model_results.get("crowd_model"),
            )
            
            # Warmup pass: compile GPU kernels and initialize buffers
            # This makes first real inference fast (eliminates compilation overhead)
            try:
                print("[MODEL WARMUP] Starting...")
                warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Run a warmup frame
                engine.process_frame(warmup_frame, show_zone_masks=False, imgsz=384)
                print("[MODEL WARMUP] Done. ")
            except Exception as e:
                print(f"[MODEL WARMUP] Warning: warmup failed (non-critical): {e}")
            
            self.ui_queue.put(("models_loaded", engine))
        except Exception as exc:
            self.ui_queue.put(("model_error", str(exc)))

    def _set_alert_visual(self, level):
        if level == "danger":
            self.status_label.configure(bg="#8B1E1E", fg="#FFFFFF")
            self.severity_var.set("CRITICAL")
            self.severity_label.configure(bg="#8B1E1E", fg="#FFFFFF")
            if not self.flash_on:
                self.alert_label.configure(bg="#8B1E1E")
            self._start_flash()
            self.alert_label.pack(fill="x", pady=(0, 10))
            self.alert_meta_label.pack(fill="x", pady=(0, 10))
        elif level == "warning":
            self.status_label.configure(bg="#8A5A14", fg="#FFFFFF")
            self.severity_var.set("WARNING")
            self.severity_label.configure(bg="#8A5A14", fg="#FFFFFF")
            self.alert_label.configure(bg="#8A5A14")
            self._stop_flash()
            self.alert_label.pack(fill="x", pady=(0, 10))
            self.alert_meta_label.pack(fill="x", pady=(0, 10))
        else:
            self.status_label.configure(bg="#1B5E20", fg="#FFFFFF")
            self.severity_var.set("SAFE")
            self.severity_label.configure(bg="#1B5E20", fg="#FFFFFF")
            self.alert_label.configure(bg="#1B5E20")
            self._stop_flash()

    def _start_flash(self):
        if self.flash_after_id is not None:
            return
        self.flash_on = False
        self._flash_tick()

    def _flash_tick(self):
        if self.last_alert_level != "danger":
            self._stop_flash()
            return

        self.flash_on = not self.flash_on
        if self.flash_on:
            self.alert_label.configure(bg="#B71C1C")
        else:
            self.alert_label.configure(bg="#6F1212")
        self.flash_after_id = self.root.after(350, self._flash_tick)

    def _stop_flash(self):
        if self.flash_after_id is not None:
            try:
                self.root.after_cancel(self.flash_after_id)
            except Exception:
                pass
            self.flash_after_id = None
        self.flash_on = False

    def _play_emergency_warning_sound(self):
        def _siren():
            pattern = [
                (950, 160),
                (1250, 160),
                (950, 160)
            ]

            for freq, duration in pattern:
                try:
                    winsound.Beep(freq, duration)
                except Exception:
                    try:
                        self.root.bell()
                    except Exception:
                        pass

        threading.Thread(target=_siren, daemon=True).start()

    def log(self, message):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{stamp}] {message}\n")
        self.log_box.see("end")

    def _validate_input_file(self, file_path):
        """Validate that the selected file is a valid image or video."""
        try:
            path_obj = Path(file_path)
            
            # Check if file exists
            if not path_obj.exists():
                messagebox.showerror("Input Error", "Selected file does not exist.")
                self.log(f"Error: File not found - {file_path}")
                return False
            
            # Check if file is readable
            if not path_obj.is_file():
                messagebox.showerror("Input Error", "Selected path is not a file.")
                return False
            
            # Check file extension
            extension = path_obj.suffix.lower()
            valid_image_ext = {".jpg", ".jpeg", ".png"}
            valid_video_ext = {".mp4"}
            valid_extensions = valid_image_ext | valid_video_ext
            
            if extension not in valid_extensions:
                messagebox.showerror(
                    "Invalid File Type",
                    f"Unsupported file format: {extension}\n\n"
                    f"Supported images: {', '.join(sorted(valid_image_ext))}\n"
                    f"Supported videos: {', '.join(sorted(valid_video_ext))}"
                )
                self.log(f"Error: Invalid file extension - {extension}")
                return False
            
            # Validate file size (optional: prevent extremely large files)
            file_size_mb = path_obj.stat().st_size / (1024 * 1024)
            if file_size_mb > 2000:  # 2GB limit
                messagebox.showwarning(
                    "Large File Warning",
                    f"File size is {file_size_mb:.1f} MB. Processing may be slow."
                )
            
            # For videos: verify it can be opened
            if extension in valid_video_ext:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    messagebox.showerror(
                        "Invalid Video File",
                        "Selected video file cannot be read. "
                        "Please ensure it is a valid video file."
                    )
                    self.log(f"Error: Cannot open video file - {file_path}")
                    return False
                
                # Try to read first frame
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    messagebox.showerror(
                        "Invalid Video File",
                        "Selected video file is corrupted or cannot be decoded. "
                        "Try converting it to MP4 format."
                    )
                    self.log(f"Error: Cannot read frames from video - {file_path}")
                    return False
            
            # For images: verify it can be opened
            elif extension in valid_image_ext:
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror(
                        "Invalid Image File",
                        "Selected image file cannot be read. "
                        "Please ensure it is a valid image file."
                    )
                    self.log(f"Error: Cannot open image file - {file_path}")
                    return False
            
            return True
        
        except Exception as e:
            messagebox.showerror("File Validation Error", f"Error validating file: {str(e)}")
            self.log(f"Error: Exception during file validation - {str(e)}")
            return False

    def select_input(self):
        self._stop_video_playback()

        path = filedialog.askopenfilename(
            title="Select image or video",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.mp4"),
                ("Supported Images", "*.jpg *.jpeg *.png"),
                ("Supported Videos", "*.mp4")
            ],
        )
        if not path:
            return

        # Validate file before processing
        if not self._validate_input_file(path):
            return

        self.current_input_path = path
        self.current_is_video = Path(path).suffix.lower() in {".mp4"}
        self.panel_zoom["input"] = 1.0
        self.panel_zoom["output"] = 1.0
        self.panel_images["input"] = None
        self.panel_images["output"] = None
        self.path_var.set(path)

        if self.current_is_video:
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if ok:
                self.current_original = frame
                self._show_on_panel(self.input_panel, frame, panel_key="input")
                self.output_panel.configure(image="", text="Labeled Preview")
                self.output_panel.image = None
            self.log("Video selected. Ready for analysis.")
            self._auto_start_processing()
        else:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Input Error", "Unable to open selected image.")
                return
            self.current_original = img
            self._show_on_panel(self.input_panel, img, panel_key="input")
            self.output_panel.configure(image="", text="Labeled Preview")
            self.output_panel.image = None
            self.log("Image selected. Ready for analysis.")
            try:
                self.alert_counter = 0
                self.alert_meta_var.set(f"Total alerts: {self.alert_counter}")
                self.last_alert_level = None
                self.last_alert_message = ""
            except Exception:
                pass
            self._auto_start_processing()

    def _auto_start_processing(self):
        if self.engine and self.current_input_path:
            self.process_selected()
        else:
            self.auto_process_pending = True

    def stop_processing(self):
        self.stop_event.set()
        self._stop_video_playback()
        self._stop_flash()
        self.log("Stop requested by operator.")

    def process_selected(self):
        self._stop_video_playback()

        if not self.engine:
            messagebox.showerror("Error", "Models are not available.")
            return
        if not self.current_input_path:
            messagebox.showinfo("Input Required", "Select image or video first.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.stop_event.clear()
        self.stop_btn.configure(state="normal")
        self.status_var.set("Processing...")
        self._set_alert_visual("normal")

        self.worker_thread = threading.Thread(target=self._worker_run, daemon=True)
        self.worker_thread.start()

    def _worker_run(self):
        try:
            if self.current_is_video:
                self._process_video()
            else:
                self._process_image()
        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))

    def _process_image(self):
        frame = cv2.imread(self.current_input_path)
        if frame is None:
            raise RuntimeError("Failed to read image for processing.")

        start = time.perf_counter()
        annotated, metrics, details = self.engine.process_frame(
            frame,
            show_zone_masks=self.show_masks_var.get(),
            imgsz=800,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        metrics["response_time_ms"] = elapsed_ms

        self.ui_queue.put(
            (
                "image_done",
                {
                    "original": frame,
                    "annotated": annotated,
                    "metrics": metrics,
                },
            )
        )

    def _process_video(self):
        cap = cv2.VideoCapture(self.current_input_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        # Reset video alert cooldown so first dangerous frame can alert immediately
        try:
            self.video_frames_since_last_alert = 999
        except Exception:
            pass
        danger_frames = 0
        zone_frames = 0
        max_people_in_danger = 0
        total_people_detected = 0
        total_people_in_danger = 0
        total_people_warning = 0
        total_latency_ms = 0.0
        confidence_values = []
        last_frame_metrics = {
            "people_detected": 0,
            "people_in_danger": 0,
            "people_on_warning": 0,
            "people_safe": 0,
            "risk_index": 0.0,
            "danger_zone_coverage_pct": 0.0,
            "yellow_zone_coverage_pct": 0.0,
        }
        preview_stride = 1
        video_imgsz = 512

        while True:
            if self.stop_event.is_set():
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            frame_start = time.perf_counter()
            annotated, metrics, _details = self.engine.process_frame(
                frame,
                show_zone_masks=self.show_masks_var.get(),
                imgsz=video_imgsz,
            )
            frame_latency_ms = (time.perf_counter() - frame_start) * 1000.0
            total_latency_ms += frame_latency_ms

            total_people_detected += metrics["people_detected"]
            total_people_in_danger += metrics["people_in_danger"]
            total_people_warning += metrics["people_on_warning"]
            if metrics.get("avg_confidence") is not None:
                confidence_values.append(float(metrics["avg_confidence"]))
            if metrics["people_in_danger"] > 0:
                danger_frames += 1
            if metrics.get("zones_detected"):
                zone_frames += 1
            max_people_in_danger = max(max_people_in_danger, metrics["people_in_danger"])
            last_frame_metrics = dict(metrics)

            if frame_idx % preview_stride == 0 or frame_idx == total_frames:
                preview_metrics = dict(metrics)
                preview_metrics["response_time_ms"] = frame_latency_ms
                self.ui_queue.put(
                    (
                        "video_progress",
                        {
                            "frame_idx": frame_idx,
                            "total_frames": total_frames,
                            "annotated": annotated,
                            "metrics": preview_metrics,
                        },
                    )
                )

        cap.release()

        processed_frames = max(frame_idx, 1)
        avg_latency_ms = total_latency_ms / processed_frames
        summary = {
            "frames_processed": frame_idx,
            "source_frames": total_frames,
            "danger_frames": danger_frames,
            "danger_frame_ratio": float(danger_frames / frame_idx) if frame_idx else 0.0,
            "max_people_in_danger": int(max_people_in_danger),
            "avg_people_detected_per_frame": float(total_people_detected / frame_idx) if frame_idx else 0.0,
            "avg_people_in_danger_per_frame": float(total_people_in_danger / frame_idx) if frame_idx else 0.0,
            "avg_people_warning_per_frame": float(total_people_warning / frame_idx) if frame_idx else 0.0,
            "response_time_ms": avg_latency_ms,
            "avg_confidence": float(np.mean(confidence_values)) if confidence_values else None,
            "zones_detected": zone_frames > 0,
            "people_detected": int(last_frame_metrics.get("people_detected", 0)),
            "people_in_danger": int(last_frame_metrics.get("people_in_danger", 0)),
            "people_on_warning": int(last_frame_metrics.get("people_on_warning", 0)),
            "people_safe": int(last_frame_metrics.get("people_safe", 0)),
            "risk_index": float(last_frame_metrics.get("risk_index", 0.0)),
            "danger_zone_coverage_pct": float(last_frame_metrics.get("danger_zone_coverage_pct", 0.0)),
            "yellow_zone_coverage_pct": float(last_frame_metrics.get("yellow_zone_coverage_pct", 0.0)),
            "total_people_detected": int(total_people_detected),
            "total_people_in_danger": int(total_people_in_danger),
            "total_people_warning": int(total_people_warning),
            "stopped_by_operator": bool(self.stop_event.is_set()),
        }

        self.ui_queue.put(
            (
                "video_done",
                {
                    "metrics": summary,
                },
            )
        )

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.ui_queue.get_nowait()

                if msg_type == "models_loaded":
                    self.engine = payload
                    self.model_ready = True
                    self.status_var.set("Models ready")
                    self.log("Models loaded successfully.")
                    if self.auto_process_pending and self.current_input_path:
                        self.auto_process_pending = False
                        self.process_selected()

                elif msg_type == "model_error":
                    self.model_ready = False
                    self.status_var.set("Model load failed")
                    messagebox.showerror("Model Error", payload)
                    self.log(f"Model error: {payload}")

                elif msg_type == "error":
                    self.status_var.set("Processing failed")
                    self._set_alert_visual("danger")
                    self.alert_var.set("SYSTEM ALERT: Processing pipeline failed")
                    self.alert_meta_var.set(f"Total alerts: {self.alert_counter}")
                    self.stop_btn.configure(state="disabled")
                    messagebox.showerror("Processing Error", payload)
                    self.log(f"Error: {payload}")

                elif msg_type == "image_done":
                    self.current_original = payload["original"]
                    self.current_annotated = payload["annotated"]
                    self.current_metrics = payload["metrics"]

                    self._show_on_panel(self.input_panel, self.current_original, panel_key="input")
                    self._show_on_panel(self.output_panel, self.current_annotated, panel_key="output")
                    self._update_metrics(self.current_metrics)
                    # Force alert evaluation for this new image (don't reuse previous message)
                    try:
                        self.last_alert_level = None
                        self.last_alert_message = ""
                    except Exception:
                        pass
                    alert_level = self._refresh_alert_status(self.current_metrics)
                    self._notify_missing_zones(self.current_metrics)

                    if alert_level == "safe":
                        self.status_var.set("Image analysis complete")
                    self.stop_btn.configure(state="disabled")
                    self.log("Image processing complete")

                elif msg_type == "video_progress":
                    frame_idx = payload["frame_idx"]
                    total_frames = payload["total_frames"]
                    self.current_annotated = payload["annotated"]
                    self.current_metrics = payload["metrics"]

                    if self.current_original is not None:
                        self._show_on_panel(self.input_panel, self.current_original, panel_key="input")
                    self._show_on_panel(self.output_panel, self.current_annotated, panel_key="output")
                    self._update_metrics(self.current_metrics)
                    # For video progress, pass source so alert logic can throttle sounds
                    alert_level = self._refresh_alert_status(self.current_metrics, source="video", frame_idx=frame_idx)

                    if alert_level == "safe":
                        if total_frames > 0:
                            self.status_var.set(f"Video processing {frame_idx}/{total_frames}")
                        else:
                            self.status_var.set(f"Video processing frame {frame_idx}")

                elif msg_type == "video_done":
                    self.current_metrics = payload["metrics"]
                    self._update_metrics(self.current_metrics)
                    alert_level = self._refresh_alert_status(self.current_metrics)
                    self._notify_missing_zones(self.current_metrics)

                    if self.current_metrics.get("stopped_by_operator"):
                        self.status_var.set("Video stopped by operator")
                        self.log("Video analysis interrupted by operator.")
                    else:
                        if alert_level == "safe":
                            self.status_var.set("Video analysis complete")
                        self.log("Video analysis completed.")

                    self.stop_btn.configure(state="disabled")

        except Empty:
            pass

        self.root.after(30, self._poll_queue)

    def _refresh_alert_status(self, metrics, source=None, frame_idx=None):
        """Update UI alert state. If called from video frames (source=='video'),
        throttle playing the danger siren to at most once every 3 frames.
        """
        danger = metrics.get("people_in_danger", 0)
        warning = metrics.get("people_on_warning", 0)
        now_str = datetime.now().strftime("%H:%M:%S")

        if danger > 0:
            message = f"CRITICAL ALERT: {danger} passenger(s) in danger zone"
            is_new_event = self.last_alert_level != "danger" or self.last_alert_message != message

            # Decide whether to play the siren
            play_siren = False
            if source == "video":
                cooldown = getattr(self, "video_frames_since_last_alert", 999)
                if is_new_event or cooldown >= 200:
                    play_siren = True
                    self.video_frames_since_last_alert = 0
                else:
                    try:
                        self.video_frames_since_last_alert = int(self.video_frames_since_last_alert) + 1
                    except Exception:
                        self.video_frames_since_last_alert = 1
            else:
                play_siren = is_new_event

            self.last_alert_level = "danger"
            self.last_alert_message = message
            self._set_alert_visual("danger")
            self.status_var.set(message)
            self.alert_var.set(message)

            if play_siren:
                try:
                    self.alert_counter += 1
                except Exception:
                    self.alert_counter = 1
                self._play_emergency_warning_sound()
                self.log(message)

            # Show last event and total alerts
            self.alert_meta_var.set(f"Last event: {now_str} | Total alerts: {self.alert_counter}")
            return "danger"
        elif warning > 0:
            message = "WARNING: Passenger(s) near restricted zone"
            is_new_event = self.last_alert_level != "warning" or self.last_alert_message != message
            self.last_alert_level = "warning"
            self.last_alert_message = message
            self._set_alert_visual("warning")
            self.status_var.set(message)
            self.alert_var.set(message)
            if is_new_event:
                try:
                    self.alert_counter += 1
                except Exception:
                    self.alert_counter = 1
                self._play_emergency_warning_sound()
                self.log(message)
            self.alert_meta_var.set(f"Last event: {now_str} | Total alerts: {self.alert_counter}")
            return "warning"
        else:
            message = "SAFE: No passenger detected in danger zone"
            self.last_alert_level = "safe"
            self.last_alert_message = message
            self._set_alert_visual("normal")
            self.alert_var.set(message)
            self.alert_meta_var.set(f"Last update: {now_str} | Total alerts: {self.alert_counter}")
            self.alert_label.pack(fill="x", pady=(0, 10))
            self.alert_meta_label.pack(fill="x", pady=(0, 10))
            return "safe"

    def _notify_missing_zones(self, metrics):
        if metrics.get("zones_detected", True):
            return

        message = (
            "Zone detection failed: no zones detected. "
            "Please ensure you are uploading a valid footage of railway platform in Klang Valley."
        )
        self.status_var.set("Zone detection failed")
        self._set_alert_visual("danger")
        messagebox.showerror("Zone Detection", message)
        self.log(message)

    def _update_metrics(self, metrics):
        def fmt_ratio(key):
            val = metrics.get(key, None)
            if val is None:
                return "N/A"
            return f"{val:.3f}"

        self.metric_vars["people_detected"].set(str(metrics.get("people_detected", 0)))
        self.metric_vars["people_in_danger"].set(str(metrics.get("people_in_danger", 0)))
        self.metric_vars["people_on_warning"].set(str(metrics.get("people_on_warning", 0)))
        self.metric_vars["people_safe"].set(str(metrics.get("people_safe", 0)))
        self.metric_vars["risk_index"].set(f"{metrics.get('risk_index', 0.0):.3f}")
        self.metric_vars["danger_zone_coverage_pct"].set(f"{metrics.get('danger_zone_coverage_pct', 0.0):.2f}%")
        self.metric_vars["yellow_zone_coverage_pct"].set(f"{metrics.get('yellow_zone_coverage_pct', 0.0):.2f}%")
        self.metric_vars["response_time_ms"].set(f"{metrics.get('response_time_ms', 0.0):.2f} ms")
        self.metric_vars["avg_confidence"].set(fmt_ratio("avg_confidence"))
        self.metric_vars["avg_zone_confidence"].set(fmt_ratio("avg_zone_confidence"))

    @staticmethod
    def _fit_for_display(bgr_image, max_w=700, max_h=500, zoom=1.0):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        w, h = img.size
        fit_scale = min(max_w / w, max_h / h)
        scale = max(0.01, fit_scale * zoom)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def _show_on_panel(self, panel, bgr_image, panel_key=None, cache_image=True):
        panel.update_idletasks()
        panel_w = panel.winfo_width()
        panel_h = panel.winfo_height()

        if panel_w <= 1 or panel_h <= 1:
            panel_w, panel_h = 700, 500

        # Keep a small padding so the rendered frame does not touch panel edges.
        max_w = max(1, panel_w - 16)
        max_h = max(1, panel_h - 16)
        zoom = self.panel_zoom.get(panel_key, 1.0) if panel_key else 1.0
        pil_img = self._fit_for_display(bgr_image, max_w=max_w, max_h=max_h, zoom=zoom)
        tk_img = ImageTk.PhotoImage(pil_img)
        panel.configure(image=tk_img, text="")
        panel.image = tk_img
        if panel_key and cache_image:
            self.panel_images[panel_key] = bgr_image

    def _on_panel_zoom(self, panel_key, event):
        delta = event.delta
        if delta == 0:
            return "break"

        zoom = self.panel_zoom.get(panel_key, 1.0)
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = max(self.zoom_min, min(self.zoom_max, zoom * factor))
        if abs(new_zoom - zoom) < 1e-6:
            return "break"

        self.panel_zoom[panel_key] = new_zoom
        image = self.panel_images.get(panel_key)
        if image is not None:
            panel = self.input_panel if panel_key == "input" else self.output_panel
            self._show_on_panel(panel, image, panel_key=panel_key, cache_image=False)

        return "break"

    def _on_panel_resize(self, panel_key, _event):
        image = self.panel_images.get(panel_key)
        if image is None:
            return
        panel = self.input_panel if panel_key == "input" else self.output_panel
        self._show_on_panel(panel, image, panel_key=panel_key, cache_image=False)

    def _start_video_playback(self, video_path, target_fps=30):
        self._stop_video_playback()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Unable to open processed video for playback: {video_path}")
            return

        self.playback_cap = cap
        self.playback_active = True
        self.log(f"Fast mode playback started at {target_fps} FPS")
        self._play_next_frame(target_fps)

    def _play_next_frame(self, target_fps=30):
        if not self.playback_active or self.playback_cap is None:
            return

        ok, frame = self.playback_cap.read()
        if not ok:
            self._stop_video_playback()
            self.log("Fast mode playback finished.")
            return

        self._show_on_panel(self.output_panel, frame, panel_key="output")
        delay_ms = max(1, int(1000 / max(1, target_fps)))
        self.playback_after_id = self.root.after(delay_ms, lambda: self._play_next_frame(target_fps))

    def _stop_video_playback(self):
        self.playback_active = False
        if self.playback_after_id is not None:
            try:
                self.root.after_cancel(self.playback_after_id)
            except Exception:
                pass
            self.playback_after_id = None

        if self.playback_cap is not None:
            try:
                self.playback_cap.release()
            except Exception:
                pass
            self.playback_cap = None


def main():
    root = tk.Tk()
    app_icon = tk.PhotoImage(file='icon.png')
    root.iconphoto(False, app_icon)
    app = PlatformSafetyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
