from ultralytics import YOLO
import cv2
import threading
from queue import Queue
from playsound import playsound
import numpy as np
import time
from datetime import timedelta
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load both models
custom_model = YOLO('models/yolov8s.pt')
coco_model = YOLO('models/coco_yolov8n.pt')

print("Custom model classes:", custom_model.names)
print("COCO model classes:", coco_model.names)

# Map COCO classes to our main activities
COCO_TO_ACTIVITY_MAP = {
    0: 'person', 39: 'eating', 40: 'eating', 41: 'eating', 42: 'eating',
    43: 'eating', 44: 'eating', 45: 'eating', 46: 'eating', 47: 'eating',
    48: 'eating', 49: 'eating', 50: 'eating', 51: 'eating', 52: 'eating',
    53: 'eating', 54: 'eating', 55: 'eating', 57: 'couch', 59: 'bed', 67: 'phone',
}

# Queues for thread communication
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)

# Tracking data
tracked_persons = {}
next_person_id = 1
ACTIVITIES = ['smoking', 'eating', 'sleeping', 'phone', 'person']

# Settings (will be saved/loaded from file)
class Settings:
    def __init__(self):
        self.time_limits = {
            'phone': 15,
            'smoking': 0,
            'eating': 0,
            'sleeping': 0,
            'person': 0
        }
        
        self.confidence_thresholds = {
            'smoking': 0.50,
            'eating': 0.30,
            'sleeping': 0.25,
            'phone': 0.25,
            'person': 0.35,
            'bed': 0.30,
            'chair': 0.30,
            'couch': 0.30
        }
        
        self.sound_on_start = []
        self.sound_on_time_limit = ['phone']
        
        self.load_settings()
    
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists('activity_settings.json'):
                with open('activity_settings.json', 'r') as f:
                    data = json.load(f)
                    self.time_limits = data.get('time_limits', self.time_limits)
                    self.confidence_thresholds = data.get('confidence_thresholds', self.confidence_thresholds)
                    self.sound_on_start = data.get('sound_on_start', self.sound_on_start)
                    self.sound_on_time_limit = data.get('sound_on_time_limit', self.sound_on_time_limit)
                    print("✓ Settings loaded from file")
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
    
    def save_settings(self):
        """Save settings to file"""
        try:
            data = {
                'time_limits': self.time_limits,
                'confidence_thresholds': self.confidence_thresholds,
                'sound_on_start': self.sound_on_start,
                'sound_on_time_limit': self.sound_on_time_limit
            }
            with open('activity_settings.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

settings = Settings()

# Enhanced tracking parameters
MAX_LOST_FRAMES = 30
IOU_THRESHOLD = 0.15
CENTER_DISTANCE_THRESHOLD = 150
detected_furniture = {'bed': [], 'chair': [], 'couch': []}

# Control Panel State
class ControlPanel:
    def __init__(self):
        self.width = 400
        self.show = True
        self.selected_activity = 'phone'
        self.dragging_slider = None
        self.activity_stats = {act: {'count': 0, 'avg_conf': 0.0} for act in ACTIVITIES}
        
    def draw(self, frame, tracked_persons):
        """Draw control panel on frame"""
        if not self.show:
            return frame
        
        h, w = frame.shape[:2]
        panel_height = h + 200 
        panel = np.zeros((panel_height, self.width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark background
        
        y = 20
        
        # Title
        cv2.putText(panel, "CONTROL PANEL", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40
        
        # Activity selector buttons
        cv2.putText(panel, "Select Activity:", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        button_width = 75
        button_height = 30
        for i, activity in enumerate(ACTIVITIES):
            if i < 3:
                row = 0
                col = i
            else:
                row = 1
                col = i - 3
                
            x = 10 + col * (button_width + 10)
            y_btn = y + row * (button_height + 10)
            
            color = (0, 120, 255) if activity == self.selected_activity else (60, 60, 60)
            cv2.rectangle(panel, (x, y_btn), (x + button_width, y_btn + button_height), color, -1)
            cv2.rectangle(panel, (x, y_btn), (x + button_width, y_btn + button_height), (255, 255, 255), 1)
            
            # Activity name
            text_size = cv2.getTextSize(activity.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x + (button_width - text_size[0]) // 2
            text_y = y_btn + (button_height + text_size[1]) // 2
            cv2.putText(panel, activity.upper(), (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Update y position after buttons
        # Buttons end at: y (original) + 40 (row 1) + 30 (height) = y + 70
        y += 85  # Reduced from 100 to save space
        
        # Selected activity settings
        activity = self.selected_activity
        cv2.line(panel, (10, y), (self.width - 10, y), (100, 100, 100), 1)
        y += 20
        
        cv2.putText(panel, f"{activity.upper()} Settings:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        y += 30
        
        # Confidence threshold slider
        conf = settings.confidence_thresholds.get(activity, 0.25)
        y = self.draw_slider(panel, "Confidence:", conf, 0.0, 1.0, 10, y, "conf")
        y += 10
        
        # Time limit slider (only for activities that support it)
        time_limit = settings.time_limits.get(activity, 0)
        y = self.draw_slider(panel, "Time Limit (s):", time_limit, 0, 300, 10, y, "time")
        y += 10
        
        # Sound toggles
        cv2.putText(panel, "Sound Alerts:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        # Sound on start checkbox
        checked_start = activity in settings.sound_on_start
        y = self.draw_checkbox(panel, "On Start", checked_start, 20, y, "sound_start")
        
        # Sound on time limit checkbox
        checked_limit = activity in settings.sound_on_time_limit
        y = self.draw_checkbox(panel, "On Time Limit", checked_limit, 20, y, "sound_limit")
        
        y += 20
        cv2.line(panel, (10, y), (self.width - 10, y), (100, 100, 100), 1)
        y += 20
        
        # Statistics
        cv2.putText(panel, "Live Statistics:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        y += 25
        
        # Count active persons per activity
        activity_counts = {act: 0 for act in ACTIVITIES}
        for pid, pdata in tracked_persons.items():
            if pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
                act = pdata['current_activity']
                activity_counts[act] += 1
        
        for act in ACTIVITIES:
            count = activity_counts[act]
            conf_val = settings.confidence_thresholds.get(act, 0.25)
            
            color = (0, 255, 0) if count > 0 else (180, 180, 180)
            cv2.putText(panel, f"{act.capitalize()}: {count}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(panel, f"conf:{conf_val:.2f}", (160, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y += 22
        
        y += 10
        cv2.line(panel, (10, y), (self.width - 10, y), (100, 100, 100), 1)
        y += 20
        
        # Save button
        cv2.rectangle(panel, (10, y), (self.width - 10, y + 35), (0, 180, 0), -1)
        cv2.rectangle(panel, (10, y), (self.width - 10, y + 35), (255, 255, 255), 2)
        cv2.putText(panel, "SAVE SETTINGS", (self.width // 2 - 70, y + 23),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Combine panel with frame
        
        w = int(w * 1.3)
        frame_resized = cv2.resize(
            frame,
            (w, panel_height),
            interpolation=cv2.INTER_LINEAR
        )


        combined = np.hstack([panel, frame_resized])
        return combined

    
    def draw_slider(self, panel, label, value, min_val, max_val, x, y, slider_id):
        """Draw a slider control"""
        cv2.putText(panel, f"{label} {value:.2f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 20
        
        slider_width = self.width - 40
        slider_x = x + 10
        
        # Slider background
        cv2.rectangle(panel, (slider_x, y), (slider_x + slider_width, y + 10), (80, 80, 80), -1)
        
        # Slider fill
        fill_width = int(((value - min_val) / (max_val - min_val)) * slider_width)
        cv2.rectangle(panel, (slider_x, y), (slider_x + fill_width, y + 10), (0, 180, 255), -1)
        
        # Slider handle
        handle_x = slider_x + fill_width
        cv2.circle(panel, (handle_x, y + 5), 8, (255, 255, 255), -1)
        cv2.circle(panel, (handle_x, y + 5), 8, (0, 180, 255), 2)
        
        return y + 25
    
    def draw_checkbox(self, panel, label, checked, x, y, checkbox_id):
        """Draw a checkbox control"""
        box_size = 20
        cv2.rectangle(panel, (x, y), (x + box_size, y + box_size), (255, 255, 255), 2)
        
        if checked:
            cv2.line(panel, (x + 4, y + 10), (x + 8, y + 16), (0, 255, 0), 2)
            cv2.line(panel, (x + 8, y + 16), (x + 16, y + 4), (0, 255, 0), 2)
        
        cv2.putText(panel, label, (x + box_size + 10, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return y + 30
    
    def handle_click(self, x, y):
        """Handle mouse click on control panel"""
        # Activity buttons (5 activities, 2 columns)
        button_width = 75
        button_height = 30
        y_start = 85
        
        for i, activity in enumerate(ACTIVITIES):
            if i < 3:
                row = 0
                col = i
            else:
                row = 1
                col = i - 3
            
            btn_x = 10 + col * (button_width + 10)
            btn_y = y_start + row * (button_height + 10)
            
            if btn_x <= x <= btn_x + button_width and btn_y <= y <= btn_y + button_height:
                self.selected_activity = activity
                return True
        
        # Check sliders
        activity = self.selected_activity
        conf_slider_y = 255  # Aligned with visual draw position
        slider_x = 20
        slider_width = self.width - 40
        
        # Confidence slider click (User requested 240-260 range)
        if slider_x <= x <= slider_x + slider_width and 240 <= y <= 265:
            self.dragging_slider = 'conf'
            self.update_active_slider(x)
            return True
            
        # Time limit slider click
        time_slider_y = 310  # Aligned with visual draw position
        if slider_x <= x <= slider_x + slider_width and time_slider_y - 15 <= y <= time_slider_y + 15:
            self.dragging_slider = 'time'
            self.update_active_slider(x)
            return True
        
        # Sound checkboxes
        checkbox_y_start = 370  # Aligned with visual draw position (was 280)
        
        # On Start checkbox (User requested 355-373 range)
        if 20 <= x <= 40 and 355 <= y <= 373:
            if activity in settings.sound_on_start:
                settings.sound_on_start.remove(activity)
            else:
                settings.sound_on_start.append(activity)
            return True
        
        # On Time Limit checkbox
        checkbox_y_start += 30  # 400
        if 20 <= x <= 40 and 386 <= y <= 404:
            if activity in settings.sound_on_time_limit:
                settings.sound_on_time_limit.remove(activity)
            else:
                settings.sound_on_time_limit.append(activity)
            return True
        
        # Save button
        save_btn_y = 635  # Moved to match visual position (was 450)
        if 10 <= x <= self.width - 10 and save_btn_y <= y <= save_btn_y + 35:
            settings.save_settings()
            print("✓ Settings saved!")
            return True
        
        return False
    
    def update_active_slider(self, x):
        """Update the currently dragged slider based on x position"""
        if not self.dragging_slider:
            return

        activity = self.selected_activity
        slider_x = 20
        slider_width = self.width - 40
        
        ratio = (x - slider_x) / slider_width
        ratio = max(0.0, min(1.0, ratio))
        
        if self.dragging_slider == 'conf':
            settings.confidence_thresholds[activity] = round(ratio, 2)
        elif self.dragging_slider == 'time':
            new_time = int(ratio * 300)
            settings.time_limits[activity] = new_time
            return True
        
        return False

control_panel = ControlPanel()

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    if not control_panel.show:
        return
    
    # Allow dragging outside panel, but other clicks must be inside
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < control_panel.width:
            control_panel.handle_click(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        control_panel.dragging_slider = None
    elif event == cv2.EVENT_MOUSEMOVE and control_panel.dragging_slider:
        control_panel.update_active_slider(x)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def calculate_center_distance(box1, box2):
    """Calculate distance between centers of two boxes"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def is_person_on_furniture(person_bbox, furniture_list):
    """Check if person overlaps with any furniture"""
    for furniture_bbox in furniture_list:
        iou = calculate_iou(person_bbox, furniture_bbox)
        if iou > 0.1:
            return True
    return False

def find_best_match(detection_bbox, tracked_persons, matched_ids):
    """Find best matching person ID"""
    best_match_id = None
    best_score = 0
    
    for pid, pdata in tracked_persons.items():
        if pid in matched_ids:
            continue
        
        frames_lost = pdata.get('frames_lost', 0)
        if frames_lost > MAX_LOST_FRAMES:
            continue
        
        iou = calculate_iou(detection_bbox, pdata['bbox'])
        center_dist = calculate_center_distance(detection_bbox, pdata['bbox'])
        dist_score = max(0, 1 - (center_dist / CENTER_DISTANCE_THRESHOLD))
        combined_score = (iou * 0.6) + (dist_score * 0.4)
        
        if frames_lost == 0:
            combined_score *= 1.2
        
        if combined_score > best_score and (iou > IOU_THRESHOLD or center_dist < CENTER_DISTANCE_THRESHOLD):
            best_score = combined_score
            best_match_id = pid
    
    return best_match_id

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def play_alert_sound(alert_type):
    """Play alert sound in separate thread"""
    try:
        if alert_type == "start":
            playsound('sound/drop.mp3')
        elif alert_type == "time_limit":
            playsound('sound/drop.mp3')
    except Exception as e:
        if alert_type == "start":
            print(f"🔊 ALERT: Activity started!")
        elif alert_type == "time_limit":
            print(f"⏰ TIME LIMIT ALERT!")

def detection_thread():
    """Run YOLO detection in separate thread"""
    global tracked_persons, next_person_id, detected_furniture
    
    frame_count = 0
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Run detections
            custom_results = custom_model(frame, imgsz=416, conf=0.25, verbose=False)
            custom_result = custom_results[0]
            
            coco_results = coco_model(frame, imgsz=416, conf=0.25, verbose=False)
            coco_result = coco_results[0]
            
            detected_furniture = {'bed': [], 'chair': [], 'couch': []}
            detections = {activity: [] for activity in ACTIVITIES}
            person_detections = []
            
            # Process custom model
            for box in custom_result.boxes:
                cls = int(box.cls)
                class_name = custom_model.names[cls]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                if class_name in ACTIVITIES:
                    required_conf = settings.confidence_thresholds.get(class_name, 0.25)
                    if conf >= required_conf:
                        detections[class_name].append({
                            'bbox': xyxy,
                            'conf': conf,
                            'source': 'custom',
                            'original_class': class_name
                        })
                        if class_name == 'person':
                            person_detections.append(xyxy)
            
            # Process COCO model
            for box in coco_result.boxes:
                cls = int(box.cls)
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                if cls in COCO_TO_ACTIVITY_MAP:
                    activity = COCO_TO_ACTIVITY_MAP[cls]
                    coco_class_name = coco_model.names[cls]
                    
                    if activity in ['bed', 'chair', 'couch']:
                        required_conf = settings.confidence_thresholds.get(activity, 0.25)
                        if conf >= required_conf:
                            detected_furniture[activity].append(xyxy)
                        continue
                    
                    required_conf = settings.confidence_thresholds.get(activity, 0.25)
                    if conf >= required_conf:
                        detections[activity].append({
                            'bbox': xyxy,
                            'conf': conf,
                            'source': 'coco',
                            'original_class': coco_class_name
                        })
                        if activity == 'person':
                            person_detections.append(xyxy)
            
            # Smart sleeping detection
            all_furniture = detected_furniture['bed'] + detected_furniture['chair'] + detected_furniture['couch']
            
            for person_bbox in person_detections:
                if is_person_on_furniture(person_bbox, all_furniture):
                    detections['sleeping'].append({
                        'bbox': person_bbox,
                        'conf': 0.8,
                        'source': 'smart',
                        'original_class': 'person+furniture'
                    })
            
            # Mark existing persons as potentially lost
            for pid in tracked_persons:
                tracked_persons[pid]['frames_lost'] = tracked_persons[pid].get('frames_lost', 0) + 1
            
            current_frame_persons = {}
            matched_ids = set()
            
            # Process detections
            for activity_name, activity_detections in detections.items():
                for detection in activity_detections:
                    det_bbox = detection['bbox']
                    det_conf = detection['conf']
                    
                    matched_id = find_best_match(det_bbox, tracked_persons, matched_ids)
                    
                    if matched_id is None:
                        person_id = next_person_id
                        next_person_id += 1
                        
                        current_frame_persons[person_id] = {
                            'bbox': det_bbox,
                            'current_activity': activity_name,
                            'detection_source': detection['source'],
                            'original_class': detection['original_class'],
                            'activities': {act: {
                                'total_time': 0,
                                'start_time': None,
                                'alerted': False,
                                'time_limit_alerted': False,
                                'conf': 0
                            } for act in ACTIVITIES},
                            'last_seen': current_time,
                            'frames_lost': 0,
                            'frame_count': frame_count
                        }
                        current_frame_persons[person_id]['activities'][activity_name]['start_time'] = current_time
                        current_frame_persons[person_id]['activities'][activity_name]['conf'] = det_conf
                    else:
                        person_id = matched_id
                        matched_ids.add(person_id)
                        
                        old_data = tracked_persons[person_id]
                        current_frame_persons[person_id] = {
                            'bbox': det_bbox,
                            'current_activity': activity_name,
                            'detection_source': detection['source'],
                            'original_class': detection['original_class'],
                            'activities': {act: old_data['activities'][act].copy() for act in ACTIVITIES},
                            'last_seen': current_time,
                            'frames_lost': 0,
                            'frame_count': old_data.get('frame_count', frame_count)
                        }
                        
                        old_activity = old_data['current_activity']
                        
                        if old_activity != activity_name and old_data['activities'][old_activity]['start_time'] is not None:
                            elapsed = current_time - old_data['activities'][old_activity]['start_time']
                            current_frame_persons[person_id]['activities'][old_activity]['total_time'] += elapsed
                            current_frame_persons[person_id]['activities'][old_activity]['start_time'] = None
                        
                        if current_frame_persons[person_id]['activities'][activity_name]['start_time'] is None:
                            current_frame_persons[person_id]['activities'][activity_name]['start_time'] = current_time
                            current_frame_persons[person_id]['activities'][activity_name]['alerted'] = False
                            current_frame_persons[person_id]['activities'][activity_name]['time_limit_alerted'] = False
                        
                        current_frame_persons[person_id]['activities'][activity_name]['conf'] = det_conf
            
            # Keep recently lost persons
            for pid, pdata in tracked_persons.items():
                if pid not in current_frame_persons and pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
                    current_frame_persons[pid] = pdata.copy()
                    current_frame_persons[pid]['activities'] = {act: pdata['activities'][act].copy() for act in ACTIVITIES}
            
            # Check time limits
            for person_id, pdata in current_frame_persons.items():
                activity = pdata['current_activity']
                activity_data = pdata['activities'][activity]
                
                if activity_data['start_time'] is not None:
                    total_time = activity_data['total_time'] + (current_time - activity_data['start_time'])
                    
                    time_limit = settings.time_limits.get(activity, 0)
                    if time_limit > 0 and total_time >= time_limit:
                        if not activity_data['time_limit_alerted']:
                            activity_data['time_limit_alerted'] = True
                            activity_data['exceeded_time_limit'] = True
            
            tracked_persons = current_frame_persons
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            if result_queue.full():
                result_queue.get()
            result_queue.put((annotated_frame, tracked_persons.copy(), current_time))

# Start detection thread
detector = threading.Thread(target=detection_thread, daemon=True)
detector.start()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Reduced from 1280 to fit standard screens

annotated_frame = None
frame_count = 0
display_tracked_persons = {}
current_time = time.time()

ACTIVITY_COLORS = {
    'smoking': (0, 0, 255),
    'eating': (0, 255, 0),
    'sleeping': (255, 0, 255),
    'phone': (0, 165, 255),
    'person': (128, 128, 128)
}

active_activities = set()

print("\n=== Interactive Activity Tracker Started ===")
print("✓ Control panel with live settings adjustment")
print("✓ Click activities to configure")
print("✓ Drag sliders to adjust thresholds")
print("✓ Toggle sound alerts with checkboxes")
print("✓ Press 'S' to save settings")
print("✓ Press 'P' to toggle panel")
print("✓ Press ESC to quit\n")

cv2.namedWindow('Smart Activity Tracker', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Smart Activity Tracker', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 2 == 0 and not frame_queue.full():
        frame_queue.put(frame.copy())
    
    if not result_queue.empty():
        annotated_frame, display_tracked_persons, current_time = result_queue.get()
        
        active_activities = set()
        for person_id, pdata in display_tracked_persons.items():
            if pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
                active_activities.add(pdata['current_activity'])
        
        for person_id, pdata in display_tracked_persons.items():
            activity = pdata['current_activity']
            activity_data = pdata['activities'][activity]
            
            if not activity_data['alerted']:
                activity_data['alerted'] = True
                
                if activity in settings.sound_on_start:
                    threading.Thread(target=play_alert_sound, args=("start",), daemon=True).start()

            if activity_data.get('exceeded_time_limit', False) and activity_data.get('time_limit_alerted', False):
                time_limit = settings.time_limits.get(activity)
                print(f"🚨 WARNING: Person ID {person_id} has been using {activity.upper()} for more than {time_limit} seconds!")
                
                if activity in settings.sound_on_time_limit:
                    threading.Thread(target=play_alert_sound, args=("time_limit",), daemon=True).start()
                
                activity_data['exceeded_time_limit'] = False
    
    if annotated_frame is not None:
        display_frame = annotated_frame.copy()
        frame_height, frame_width = display_frame.shape[:2]
        
        # Display active activities on screen
        activity_panel_y = 30
        title_text = "Active Activities:"
        title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        title_x = frame_width - title_size[0] - 10
        
        cv2.putText(display_frame, title_text, (title_x, activity_panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        activity_panel_y += 30
        
        if active_activities:
            for activity in sorted(active_activities):
                color = ACTIVITY_COLORS.get(activity, (255, 255, 255))
                activity_text = f"{activity.upper()}"
                activity_size = cv2.getTextSize(activity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                activity_x = frame_width - activity_size[0] - 10
                
                cv2.putText(display_frame, activity_text, (activity_x, activity_panel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                activity_panel_y += 25
        else:
            none_text = "None"
            none_size = cv2.getTextSize(none_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            none_x = frame_width - none_size[0] - 10
            
            cv2.putText(display_frame, none_text, (none_x, activity_panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        # Draw person boxes
        for person_id, pdata in display_tracked_persons.items():
            frames_lost = pdata.get('frames_lost', 0)
            
            if frames_lost > 0:
                continue
            
            bbox = pdata['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            activity = pdata['current_activity']
            activity_data = pdata['activities'][activity]
            color = ACTIVITY_COLORS.get(activity, (255, 255, 255))
            
            total_time = activity_data['total_time']
            if activity_data['start_time'] is not None:
                current_session = current_time - activity_data['start_time']
                total_time += current_session
            
            time_limit = settings.time_limits.get(activity, 0)
            warning_text = ""
            if time_limit > 0:
                if total_time >= time_limit:
                    color = (0, 0, 255)
                    warning_text = "⚠️ TIME LIMIT!"
                elif total_time >= time_limit * 0.8:
                    color = (0, 165, 255)
                    remaining = int(time_limit - total_time)
                    warning_text = f"⏰ {remaining}s left"
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            # Build info text
            info_text = [f"{activity.upper()}"]
            
            conf_val = activity_data.get('conf', 0)
            info_text.append(f"Conf: {conf_val:.2f}")
            
            if activity in settings.sound_on_time_limit and time_limit > 0:
                time_str = format_time(total_time)
                info_text.append(f"Time: {time_str}")
            
            if warning_text:
                info_text.append(warning_text)
            
            # Draw text at TOP CENTER of box
            box_center_x = (x1 + x2) // 2
            text_y = y1 + 20

            for text in info_text:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = box_center_x - text_size[0] // 2
                
                cv2.rectangle(display_frame,
                            (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5),
                            color, -1)
                
                cv2.putText(display_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                text_y += text_size[1] + 15
        
        # Add control panel
        display_frame = control_panel.draw(display_frame, display_tracked_persons)
        cv2.imshow('Smart Activity Tracker', display_frame)
    else:
        display_frame = control_panel.draw(frame, {})
        cv2.imshow('Smart Activity Tracker', display_frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        frame_queue.put(None)
        break
    elif key == ord('p') or key == ord('P'):  # Toggle panel
        control_panel.show = not control_panel.show
    elif key == ord('s') or key == ord('S'):  # Save settings
        settings.save_settings()
        print("✓ Settings saved!")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("Final Activity Statistics:")
print("="*50)
for person_id, pdata in tracked_persons.items():
    if pdata.get('frames_lost', 0) <= MAX_LOST_FRAMES:
        print(f"\nPerson ID {person_id}:")
        for activity, data in pdata['activities'].items():
            total = data['total_time']
            if pdata['current_activity'] == activity and data['start_time'] is not None:
                total += time.time() - data['start_time']
            if total > 0:
                time_str = format_time(total)
                limit = settings.time_limits.get(activity, 0)
                limit_str = f" (Limit: {limit}s)" if limit > 0 else ""
                exceeded = " ⚠️ EXCEEDED" if limit > 0 and total >= limit else ""
                print(f"  {activity.capitalize()}: {time_str}{limit_str}{exceeded}")