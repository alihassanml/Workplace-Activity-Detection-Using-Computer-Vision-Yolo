# Workplace Activity Detection Video Analytics System

## 📄 Overview  
This repository contains a custom AI‑powered video analytics system for workplace monitoring.  
It can detect predefined activities in CCTV / video footage such as:  
- Mobile phone usage  
- Sleeping at workstation  
- Eating at desk  
- Smoking (indoor/outdoor)  
- Clock in / Clock out gestures  

The goal: turn regular office cameras into a smart, automated watchdog — giving you visibility, logs, and reports.  

---

## ⚙️ Tech Stack & Tools  

- **Python** — core language for scripts and backend  
- **Object / Activity Detection Models** — e.g. YOLOv8 (or other) + OpenCV + optionally MediaPipe for pose analysis  
- **Video / Frame Handling** — OpenCV for frame extraction and manipulation  
- **Web Backend** — FastAPI (or Flask) for REST API endpoints  
- **Containerization** — Docker for deployment  
- **(Optional) Frontend Dashboard** — simple web UI (if implemented)  

---

## 📂 Repository Structure  

```

WorkplaceAI/
│
├── data/
│   ├── raw/
│   ├── frames/
│   ├── labels/
│   └── dataset/         # YOLO formatted (train / val / test)
│
├── models/
│   ├── pretrained/
│   └── trained/
│
├── scripts/             # data processing & training scripts
│
├── backend/             # API + detection pipeline
│
├── frontend/ (optional) # dashboard UI
│
├── reports/             # logs / screenshots of detections
│
└── docs/                # documentation, guides

````

Feel free to modify — this is a simple, clean starting point.  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- Pip  
- (Optional) Docker & Docker‑Compose  

### Installation & Setup  

1. Clone the repo  
```bash
git clone https://github.com/alihassanml/Video-Analytics.git  
cd Video-Analytics  
````

2. Create & activate virtual environment

```bash
python -m venv venv  
source venv/bin/activate   # (Linux / macOS)  
```

3. Install dependencies

```bash
pip install -r requirements.txt  
```

4. (Optional) If using Docker — build the container

```bash
docker build -t workplace-ai .  
```

---

## 🧪 How to Use / Example Workflow

1. Put raw video(s) into `data/raw/`.
2. Run frame extraction script to generate frames.
3. Label images (bounding‑boxes etc.) and prepare YOLO dataset.
4. Train the detection model using training script.
5. Use the trained model to run detection on new video / live feed — backend will flag events.
6. (Optional) Launch web dashboard or export logs/reports.

---

## ✅ Current Features & What’s Pending

**Implemented / Planned**

* [ ] Mobile phone usage detection
* [ ] Sleeping detection
* [ ] Eating detection
* [ ] Smoking detection
* [ ] Clock in/out detection
* [ ] Logging & report generation
* [ ] Web API for detection & retrieval
* [ ] Docker‑based deployment

---

## 📚 Why This Project? Motivation

Manual surveillance and periodic checks are unreliable.
This system offers a **consistent, unbiased, automatic** way to monitor workplace behavior — reducing human error and ensuring policy compliance.
It can help organizations maintain discipline, safety, and productivity — without invasive, manual monitoring.

---

## 🤝 Contribution & Future Work

Feel free to fork the repo, open issues, or submit pull requests.
Planned enhancements:

* Real‑time live video stream processing
* Notification system for flagged events
* Better UI / Dashboard
* More robust detection (multi‑angle, lighting variation)

---

## 📝 License & Contact

This project is currently under MIT license.
If you have any questions or suggestions — open an issue or contact me directly.

