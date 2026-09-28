# Vehicle Detection and Tracking System Using YOLOv8 and OpenCV

An Academic B.Tech Computer Vision and Deep Learning Capstone Project.

---

## 📌 Executive Summary

**Vehicle Track AI** is a production-grade web application built using **Python, Django, Ultralytics YOLOv8, ByteTrack, and OpenCV**. It addresses the challenge of real-time multi-class vehicle detection, object tracking, and non-duplicate vehicle counting from traffic surveillance videos and live webcam feeds.

Unlike naive frame-by-frame detection systems that count the same stationary or moving vehicle repeatedly on every frame, this system integrates **ByteTrack multi-object tracking** to assign persistent tracking IDs (`track_id`) to individual vehicles across sequential video frames. A unique counting filter ensures that each vehicle is counted **exactly once** per session.

---

## ✨ Features

- 🏎️ **Multi-Class Vehicle Detection**: Detects **Car** (COCO ID 2), **Motorcycle** (COCO ID 3), **Bus** (COCO ID 5), and **Truck** (COCO ID 7).
- 🎯 **ByteTrack Trajectory Tracking**: Maintains track IDs across occlusions, camera motion, and object overlaps.
- 🚫 **Duplicate Prevention**: Session-safe hashing prevents frame count inflation.
- 📹 **Traffic Video Processing**: Upload common video formats (`.mp4`, `.avi`, `.mov`, `.mkv`), process frames sequentially, and render bounding box overlays.
- 📊 **Results Dashboard**: Displays category breakdown metrics, processing duration, FPS, and an embedded HTML5 video player for annotated output videos.
- 🎥 **Live Webcam Tracking**: Stream real-time vehicle detection and tracking from local USB/laptop webcams via MJPEG stream with a live counter reset API.
- 🔐 **Authentication & History**: User registration, login, logout, and persistent SQLite user history archives.
- 🛠️ **Admin Management**: Django Admin panel for monitoring users and detection logs.

---

## 🛠️ Technology Stack

| Component | Technology Used | Description |
|---|---|---|
| **Backend Framework** | Python 3.10+, Django 5.x | Web application routing, ORM models, views |
| **Object Detection** | Ultralytics YOLOv8 (`yolov8s.pt`) | Deep learning single-stage object detector |
| **Object Tracking** | ByteTrack | Multi-object tracking algorithm via Ultralytics |
| **Computer Vision** | OpenCV (`opencv-python`) | Frame decoding, HUD overlay drawing, VideoWriter |
| **Database** | SQLite 3 | Embedded transactional SQL database |
| **Frontend UI** | HTML5, Vanilla CSS3, JS, Django Templates | Responsive dark glassmorphism design system |

---

## 🏗️ System Architecture

```
User (Browser)
   │
   ├──> Django Web Application
   │       ├── Authentication System (Login / Register)
   │       ├── Video Upload Interface
   │       ├── Results Dashboard
   │       └── Live Webcam Tracker
   │
   └──> Video Processing Pipeline
           ├── OpenCV VideoCapture (Frame decoding)
           ├── YOLOv8s Detection (Car, Bus, Truck, Motorcycle)
           ├── ByteTrack Multi-Object Tracking (Assign Track IDs)
           ├── Duplicate Prevention Filter (class_name, track_id)
           ├── OpenCV Frame Renderer (HUD overlay & bounding boxes)
           └── VideoWriter (.mp4 output generation)
                   │
                   └──> SQLite Database (VideoRecord ORM)
```

---

## 📁 Directory Structure

```
RC/
├── manage.py                   # Django CLI entrypoint
├── requirements.txt            # Dependency configuration
├── README.md                   # Complete documentation & Viva Guide
├── .gitignore                  # Git ignore rules
├── .env.example                # Sample environment variables
│
├── project_config/             # Django core settings & routing
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── detection/                  # Primary Django application
│   ├── models.py               # VideoRecord SQLite model
│   ├── views.py                # View handlers (upload, results, live stream)
│   ├── forms.py                # Upload & auth form validations
│   ├── urls.py                 # Route mappings
│   ├── admin.py                # Django admin registration
│   ├── tests.py                # Automated unit test suite (TC01-TC10)
│   │
│   ├── services/               # Computer vision services
│   │   ├── __init__.py
│   │   └── video_processor.py  # YOLOv8 + ByteTrack processing engine
│   │
│   ├── templates/detection/    # HTML5 Django templates
│   │   ├── base.html           # Master layout with navbar & footer
│   │   ├── home.html           # Landing page with project summary
│   │   ├── about.html          # Academic project documentation
│   │   ├── upload.html         # Drag-and-drop video upload form
│   │   ├── results.html        # Multi-class count results dashboard
│   │   ├── live_tracker.html   # Live webcam tracking interface
│   │   ├── history.html        # Video analysis archive
│   │   ├── dashboard.html      # User analytics dashboard
│   │   ├── login.html          # User login
│   │   └── register.html       # User registration
│   │
│   └── static/detection/
│       └── css/style.css       # Custom modern dark UI styling
│
├── media/                      # Storage for media files
│   ├── uploads/                # Original uploaded videos
│   └── processed/              # Annotated output videos
└── db.sqlite3                  # Initialized SQLite database
```

---

## 🚀 Quick Start Guide (How to Run)

Follow these explicit commands to run the project locally on Windows:

### 1. Open Terminal & Navigate to Project Directory
```cmd
cd c:\Users\ragar\OneDrive\Desktop\RC
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 4. Run Database Migrations
```cmd
python manage.py migrate
```

### 5. Initialize Seed Demo Accounts (Superuser & Student)
```cmd
python manage.py init_demo
```
*Creates Superuser: `admin` / Password: `adminpassword`*
*Creates Demo User: `student` / Password: `studentpassword`*

### 6. Start Development Server
```cmd
python manage.py runserver
```

### 7. Access Web Application
Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

---

## 🧪 Automated Testing

Run the built-in Django automated test suite covering test cases `TC01` to `TC10`:

```cmd
python manage.py test detection
```

---

## 💡 Viva Cheat Sheet (Q&A for Project Defense)

### Q1: What is YOLOv8 and why was it selected for this project?
**Answer:** YOLOv8 (You Only Look Once, version 8) developed by Ultralytics is a state-of-the-art single-stage object detector. It formulates object detection as a single regression task, directly predicting bounding box coordinates and class probabilities in a single forward pass through the neural network. We use `yolov8s.pt` (small variant) because it delivers an optimal balance of high detection speed (FPS) and accuracy suitable for execution on standard CPU/GPU laptops.

### Q2: Why is object tracking essential for vehicle counting?
**Answer:** In video processing, a vehicle remains visible across dozens or hundreds of consecutive video frames. If we simply counted detections on every frame, a single car passing through a 300-frame video segment would be counted 300 times. Tracking assigns a persistent `track_id` to each vehicle trajectory across time.

### Q3: What object tracking algorithm is used, and how does ByteTrack work?
**Answer:** The system uses **ByteTrack**. Unlike classical trackers that throw away low-confidence detection boxes, ByteTrack retains both high-confidence and low-confidence boxes. It pairs high-confidence detections first, and then associates low-confidence detections with unconfirmed tracks, significantly improving tracking accuracy during temporary occlusions (e.g. when a car passes behind a bus).

### Q4: How is duplicate counting prevented in the code?
**Answer:** During video frame iteration, the system maintains a set of unique tuples: `(vehicle_class_name, track_id)`. When a tracked vehicle is detected in a frame:
1. We check if `(cls_name, track_id)` exists in our `tracked_vehicles` set.
2. If it is **not present**, it is added to the set and the category counter for that vehicle class is incremented by 1.
3. If it is **already present**, the count remains unchanged.

### Q5: What vehicle classes are supported?
**Answer:** We restrict detection to COCO dataset indices corresponding to road traffic:
- **Car** (Class ID: 2)
- **Motorcycle** (Class ID: 3)
- **Bus** (Class ID: 5)
- **Truck** (Class ID: 7)

### Q6: How does the Live Webcam mode work?
**Answer:** The live webcam mode utilizes OpenCV `cv2.VideoCapture(0)` to grab frames from the system camera, runs `yolov8s.pt` tracking on each frame, draws bounding boxes and live HUD counters, and streams the frames to the web page via Django `StreamingHttpResponse` using MJPEG (multipart/x-mixed-replace) format.

---

## 🔮 Limitations & Future Enhancements

- **Limitations**: Severe rain/fog, dark night conditions without illumination, or severe camera shake can affect tracking ID consistency.
- **Future Enhancements**:
  - Automatic License Plate Recognition (ALPR).
  - Vehicle speed estimation based on pixel calibration.
  - Traffic congestion density heatmaps.
  - Integration with external RTSP IP surveillance camera streams.
