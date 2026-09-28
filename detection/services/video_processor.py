import os
import sys
import time
import uuid
import logging
import cv2
import numpy as np
import torch
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

# COCO Vehicle Class ID Mapping
VEHICLE_CLASS_MAP = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

CLASS_COLORS = {
    'car': (255, 140, 0),        # Vibrant Deep Blue / Amber (BGR)
    'bus': (0, 200, 100),       # Emerald Green
    'truck': (200, 50, 255),    # Neon Violet
    'motorcycle': (0, 215, 255) # Bright Yellow/Gold
}

_yolo_model = None
_yolo_device = None

def get_device():
    """
    Automatically detects CUDA GPU availability; falls back to CPU.
    """
    global _yolo_device
    if _yolo_device is None:
        if torch.cuda.is_available():
            _yolo_device = 0  # GPU index 0
        else:
            _yolo_device = 'cpu'
    return _yolo_device

def get_yolo_model():
    """
    Singleton loader for YOLOv8 model to avoid reloading on every request.
    Uses yolov8s.pt pre-trained model.
    """
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = getattr(settings, 'YOLO_MODEL_PATH', 'yolov8s.pt')
            device = get_device()
            logger.info(f"Loading YOLOv8 model from '{model_path}' on device '{device}'...")
            _yolo_model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded successfully on device '{device}'.")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise RuntimeError(f"YOLOv8 Model initialization failed: {str(e)}")
    return _yolo_model


class VideoProcessor:
    """
    Core Computer Vision Processing Pipeline:
    1. Read input video frames sequentially using OpenCV.
    2. Perform vehicle detection & object tracking using YOLOv8 + ByteTrack.
    3. Filter detections to target vehicle classes: Car, Bus, Truck, Motorcycle.
    4. Track vehicle IDs across frames to prevent duplicate counting.
    5. Render bounding boxes, class labels, tracking IDs, and HUD count overlays.
    6. Write annotated frames to media/processed output video file.
    7. Save precise breakdown counts to SQLite database.
    """

    def __init__(self, input_path, output_filename=None):
        self.input_path = str(input_path)
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input video file not found at: {self.input_path}")

        # Ensure processed output directory exists
        processed_dir = Path(settings.MEDIA_ROOT) / 'processed'
        os.makedirs(processed_dir, exist_ok=True)

        if not output_filename:
            output_filename = f"processed_{uuid.uuid4().hex[:12]}.mp4"
        
        self.output_filename = output_filename
        self.output_path = str(processed_dir / self.output_filename)
        self.relative_output_path = f"processed/{self.output_filename}"

        self.tracked_vehicles = set()  # Stores tuples: (class_name, track_id)
        self.category_counts = {
            'car': 0,
            'bus': 0,
            'truck': 0,
            'motorcycle': 0
        }

    def process_video(self, progress_callback=None):
        """
        Processes the input video file, runs tracking, draws overlays,
        and saves the output video with high throughput and low memory footprint.
        """
        start_time = time.time()
        model = get_yolo_model()
        device = get_device()

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {self.input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Fallback default FPS
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        # Use mp4v fourcc codec for widest Windows/browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            # Fallback codec attempt if mp4v fails
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("OpenCV VideoWriter failed to initialize output video file.")

        frame_index = 0
        try:
            with torch.no_grad():
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    frame_index += 1

                    # Optimized YOLOv8 ByteTrack tracking call:
                    # - imgsz=640 standard inference resolution for high speed
                    # - device=device (CUDA GPU if available, else CPU)
                    # - persist=True maintains ByteTrack trajectory state
                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[2, 3, 5, 7],
                        imgsz=640,
                        device=device,
                        verbose=False
                    )

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            # Vectorized batch transfer of tensor attributes to CPU lists
                            cls_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
                            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls_ids)
                            coords = boxes.xyxy.int().cpu().tolist() if boxes.xyxy is not None else []

                            for cls_id, track_id, (x1, y1, x2, y2) in zip(cls_ids, track_ids, coords):
                                cls_name = VEHICLE_CLASS_MAP.get(cls_id, 'vehicle')

                                # Unique Counting Logic:
                                # A vehicle is counted ONCE per unique (cls_name, track_id) pair.
                                if track_id is not None:
                                    track_key = (cls_name, track_id)
                                    if track_key not in self.tracked_vehicles:
                                        self.tracked_vehicles.add(track_key)
                                        if cls_name in self.category_counts:
                                            self.category_counts[cls_name] += 1

                                color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                                id_str = f" ID:{track_id}" if track_id is not None else ""
                                label = f"{cls_name.upper()}{id_str}"

                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color, -1)
                                cv2.putText(
                                    frame,
                                    label,
                                    (x1 + 5, max(15, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA
                                )

                    # Draw Top HUD Banner (Sub-region ROI blending)
                    self._draw_hud_banner(frame)

                    # Write processed frame
                    writer.write(frame)

                    if progress_callback and (frame_index % 5 == 0 or frame_index == total_frames):
                        pct = min(100, int((frame_index / total_frames) * 100))
                        try:
                            progress_callback(frame_index, total_frames, pct, self.category_counts.copy())
                        except Exception as cb_err:
                            logger.warning(f"Progress callback error: {cb_err}")

        finally:
            cap.release()
            writer.release()

        processing_time = round(time.time() - start_time, 2)
        overall_total = sum(self.category_counts.values())

        if not os.path.exists(self.output_path) or os.path.getsize(self.output_path) == 0:
            raise RuntimeError("Processing failed: Output video file was not generated or is empty.")

        return {
            'relative_output_path': self.relative_output_path,
            'output_path': self.output_path,
            'filename': self.output_filename,
            'car_count': self.category_counts['car'],
            'bus_count': self.category_counts['bus'],
            'truck_count': self.category_counts['truck'],
            'motorcycle_count': self.category_counts['motorcycle'],
            'overall_count': overall_total,
            'fps': round(fps, 2),
            'total_frames': frame_index,
            'processing_time_seconds': processing_time,
        }

    def _draw_hud_banner(self, frame):
        """
        Draws a semi-transparent dark HUD banner across the top ROI of the video frame
        using fast sub-region blending without full frame copying.
        """
        h, w, _ = frame.shape
        banner_h = min(45, h)
        
        # Sub-region ROI blend (avoids full frame.copy())
        sub = frame[0:banner_h, 0:w]
        bg = np.full_like(sub, (15, 23, 42))
        cv2.addWeighted(bg, 0.75, sub, 0.25, 0, sub)

        # Header Text
        hud_text = (
            f"CARS: {self.category_counts['car']} | "
            f"BUSES: {self.category_counts['bus']} | "
            f"TRUCKS: {self.category_counts['truck']} | "
            f"MOTORCYCLES: {self.category_counts['motorcycle']} | "
            f"TOTAL: {sum(self.category_counts.values())}"
        )
        
        cv2.putText(
            frame,
            hud_text,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 240, 255),
            2,
            cv2.LINE_AA
        )


class LiveWebcamTracker:
    """
    Live streaming handler using OpenCV VideoCapture for local webcam feed.
    """

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.tracked_ids = set()
        self.counts = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}

    def reset_counts(self):
        self.tracked_ids.clear()
        self.counts = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}

    def generate_frames(self):
        model = get_yolo_model()
        device = get_device()
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank_frame,
                "Webcam Unavailable / Access Denied",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            _, jpeg = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            return

        try:
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[2, 3, 5, 7],
                        imgsz=640,
                        device=device,
                        verbose=False
                    )

                    if results and len(results) > 0:
                        boxes = results[0].boxes
                        if boxes is not None and len(boxes) > 0:
                            cls_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
                            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls_ids)
                            coords = boxes.xyxy.int().cpu().tolist() if boxes.xyxy is not None else []

                            for cls_id, track_id, (x1, y1, x2, y2) in zip(cls_ids, track_ids, coords):
                                cls_name = VEHICLE_CLASS_MAP.get(cls_id, 'vehicle')

                                if track_id is not None:
                                    key = (cls_name, track_id)
                                    if key not in self.tracked_ids:
                                        self.tracked_ids.add(key)
                                        if cls_name in self.counts:
                                            self.counts[cls_name] += 1

                                color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                id_str = f" ID:{track_id}" if track_id is not None else ""
                                label = f"{cls_name.upper()}{id_str}"
                                
                                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color, -1)
                                cv2.putText(frame, label, (x1 + 5, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # HUD Banner Sub-region ROI blend
                    h, w, _ = frame.shape
                    banner_h = min(40, h)
                    sub = frame[0:banner_h, 0:w]
                    bg = np.full_like(sub, (15, 23, 42))
                    cv2.addWeighted(bg, 0.75, sub, 0.25, 0, sub)

                    text = f"LIVE CARS:{self.counts['car']} | BUSES:{self.counts['bus']} | TRUCKS:{self.counts['truck']} | MOTORS:{self.counts['motorcycle']} | TOTAL:{sum(self.counts.values())}"
                    cv2.putText(frame, text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        finally:
            cap.release()
