from ultralytics import YOLO
import cv2
import threading
from queue import Queue
from playsound import playsound
import numpy as np
import time
from datetime import timedelta

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = 'models/yolov8s.pt'
model = YOLO(path)
print("Model classes:", model.names)

# Queues for thread communication
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)

# Tracking data
tracked_persons = {}  
# Structure: {person_id: {
#     'bbox': [x,y,w,h], 
#     'current_activity': 'phone',
#     'activities': {
#         'phone': {'total_time': 0, 'start_time': timestamp, 'alerted': False, 'time_limit_alerted': False},
#         'smoking': {'total_time': 0, 'start_time': None, 'alerted': False, 'time_limit_alerted': False},
#         ...
#     },
#     'last_seen': frame_count
# }}

next_person_id = 1
ACTIVITIES = ['smoking', 'eating', 'sleeping', 'phone']

# Time limits for alerts (in seconds)
TIME_LIMITS = {
    'phone': 15,      # 1 minute for phone usage
    'smoking': None,    # Optional: 30 seconds for smoking
    'eating': None,   # No limit for eating
    'sleeping': None  # No limit for sleeping
}

ALERT_SOUND_ACTIVITIES = ['phone'] 

CONFIDENCE_THRESHOLDS = {
    'smoking': 0.50,   # 50% confidence for smoking
    'eating': 0.25,    # 25% confidence for eating
    'sleeping': 0.25,  # 25% confidence for sleeping
    'phone': 0.25      # 25% confidence for phone
}
PLAY_SOUND_ON_START = [] 
PLAY_SOUND_ON_TIME_LIMIT = ['phone'] 

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def play_alert_sound(alert_type):
    """Play alert sound in separate thread"""
    try:
        if alert_type == "start":
            playsound('sound/drop.mp3')  # Sound when activity starts
        elif alert_type == "time_limit":
            playsound('sound/drop.mp3')   # Drop sound when time limit exceeded
    except Exception as e:
        if alert_type == "start":
            print(f"🔊 ALERT: Activity started!")
        elif alert_type == "time_limit":
            print(f"⏰ TIME LIMIT ALERT: Drop sound!")

def detection_thread():
    """Run YOLO detection in separate thread"""
    global tracked_persons, next_person_id
    
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:  # Stop signal
                break
            
            results = model(frame, imgsz=416, conf=0.25, verbose=False)
            result = results[0]
            
            current_time = time.time()
            
            # Extract detections by class
            detections = {activity: [] for activity in ACTIVITIES}
            
            for box in result.boxes:
                cls = int(box.cls)
                class_name = model.names[cls]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                
                if class_name in ACTIVITIES:
                    # Check if confidence meets the threshold for this activity
                    required_conf = CONFIDENCE_THRESHOLDS.get(class_name, 0.25)
                    if conf >= required_conf:
                        detections[class_name].append({'bbox': xyxy, 'conf': conf})
            
            # Update tracked persons
            current_frame_persons = {}
            matched_ids = set()
            
            # Process each activity detection
            for activity_name, activity_detections in detections.items():
                for detection in activity_detections:
                    det_bbox = detection['bbox']
                    
                    # Try to match with existing person
                    matched_id = None
                    max_iou = 0.3  # Minimum IoU threshold
                    
                    for pid, pdata in tracked_persons.items():
                        if pid in matched_ids:
                            continue
                        
                        iou = calculate_iou(det_bbox, pdata['bbox'])
                        if iou > max_iou:
                            max_iou = iou
                            matched_id = pid
                    
                    # Assign ID
                    if matched_id is None:
                        person_id = next_person_id
                        next_person_id += 1
                        
                        # Initialize new person
                        current_frame_persons[person_id] = {
                            'bbox': det_bbox,
                            'current_activity': activity_name,
                            'activities': {act: {
                                'total_time': 0, 
                                'start_time': None, 
                                'alerted': False,
                                'time_limit_alerted': False
                            } for act in ACTIVITIES},
                            'last_seen': current_time
                        }
                        # Start timing for this activity
                        current_frame_persons[person_id]['activities'][activity_name]['start_time'] = current_time
                    else:
                        person_id = matched_id
                        matched_ids.add(person_id)
                        
                        # Copy existing person data
                        old_data = tracked_persons[person_id]
                        current_frame_persons[person_id] = {
                            'bbox': det_bbox,
                            'current_activity': activity_name,
                            'activities': {act: old_data['activities'][act].copy() for act in ACTIVITIES},
                            'last_seen': current_time
                        }
                        
                        # Update activity timing
                        old_activity = old_data['current_activity']
                        
                        # Stop old activity timer if different
                        if old_activity != activity_name and old_data['activities'][old_activity]['start_time'] is not None:
                            elapsed = current_time - old_data['activities'][old_activity]['start_time']
                            current_frame_persons[person_id]['activities'][old_activity]['total_time'] += elapsed
                            current_frame_persons[person_id]['activities'][old_activity]['start_time'] = None
                        
                        # Start new activity timer
                        if current_frame_persons[person_id]['activities'][activity_name]['start_time'] is None:
                            current_frame_persons[person_id]['activities'][activity_name]['start_time'] = current_time
                            current_frame_persons[person_id]['activities'][activity_name]['alerted'] = False
                            current_frame_persons[person_id]['activities'][activity_name]['time_limit_alerted'] = False
            
            # Check time limits for ongoing activities
            for person_id, pdata in current_frame_persons.items():
                activity = pdata['current_activity']
                activity_data = pdata['activities'][activity]
                
                if activity_data['start_time'] is not None:
                    # Calculate total time for this activity
                    total_time = activity_data['total_time'] + (current_time - activity_data['start_time'])
                    
                    # Check if time limit exceeded
                    time_limit = TIME_LIMITS.get(activity)
                    if time_limit is not None and total_time >= time_limit:
                        if not activity_data['time_limit_alerted']:
                            activity_data['time_limit_alerted'] = True
                            activity_data['exceeded_time_limit'] = True
            
            # Update tracked persons
            tracked_persons = current_frame_persons
            
            # Send results
            annotated_frame = result.plot()
            
            if result_queue.full():
                result_queue.get()
            result_queue.put((annotated_frame, tracked_persons.copy(), current_time))

# Start detection thread
detector = threading.Thread(target=detection_thread, daemon=True)
detector.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

annotated_frame = None
frame_count = 0
display_tracked_persons = {}
current_time = time.time()

# Activity colors
ACTIVITY_COLORS = {
    'smoking': (0, 0, 255),      # Red
    'eating': (0, 255, 0),        # Green
    'sleeping': (255, 0, 255),    # Magenta
    'phone': (0, 165, 255)        # Orange
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Send every 4th frame to detection thread
    if frame_count % 2 == 0 and not frame_queue.full():
        frame_queue.put(frame.copy())
    
    # Get latest detection result
    if not result_queue.empty():
        annotated_frame, display_tracked_persons, current_time = result_queue.get()
        
        # Check for new activities and alerts
        for person_id, pdata in display_tracked_persons.items():
            activity = pdata['current_activity']
            activity_data = pdata['activities'][activity]
            
            # Alert when activity starts
            # Alert when activity starts (only for specific activities)
            # Alert when activity starts (play sound immediately for certain activities)
            if not activity_data['alerted']:
                print(f"⚠️ Person ID {person_id} started: {activity.upper()}")
                activity_data['alerted'] = True
                
                # Play sound immediately for activities in PLAY_SOUND_ON_START
                if activity in PLAY_SOUND_ON_START:
                    print(f"🔊 Playing start sound for {activity}")
                    threading.Thread(target=play_alert_sound, args=("start",), daemon=True).start()

            # Alert when time limit exceeded (play sound after time limit)
            if activity_data.get('exceeded_time_limit', False) and activity_data.get('time_limit_alerted', False):
                time_limit = TIME_LIMITS.get(activity)
                print(f"🚨 WARNING: Person ID {person_id} has been using {activity.upper()} for more than {time_limit} seconds!")
                
                # Play sound for activities in PLAY_SOUND_ON_TIME_LIMIT
                if activity in PLAY_SOUND_ON_TIME_LIMIT:
                    print(f"🔊 Playing time limit sound for {activity}")
                    threading.Thread(target=play_alert_sound, args=("time_limit",), daemon=True).start()
                
                # Mark as handled so we don't spam alerts
                activity_data['exceeded_time_limit'] = False
    
    # Display
    if annotated_frame is not None:
        display_frame = annotated_frame.copy()
        
        # Draw person tracking info
        for person_id, pdata in display_tracked_persons.items():
            bbox = pdata['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            activity = pdata['current_activity']
            activity_data = pdata['activities'][activity]
            color = ACTIVITY_COLORS.get(activity, (255, 255, 255))
            
            # Calculate current total time for ongoing activity
            total_time = activity_data['total_time']
            if activity_data['start_time'] is not None:
                current_session = current_time - activity_data['start_time']
                total_time += current_session
            
            time_str = format_time(total_time)
            
            # Check if time limit is being approached or exceeded
            time_limit = TIME_LIMITS.get(activity)
            warning_text = ""
            if time_limit is not None:
                if total_time >= time_limit:
                    color = (0, 0, 255)  # Red - time limit exceeded
                    warning_text = "⚠️ TIME LIMIT!"
                elif total_time >= time_limit * 0.8:  # 80% of limit
                    color = (0, 165, 255)  # Orange - approaching limit
                    remaining = int(time_limit - total_time)
                    warning_text = f"⏰ {remaining}s left"
            
            # Draw person box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw info background
            info_text = [
                f"ID: {person_id}",
                f"{activity.upper()}",
                f"Time: {time_str}",
                f"Conf: {int(activity_data.get('conf', 0) * 100)}%" 
            ]
            
            if warning_text:
                info_text.append(warning_text)
            
            y_offset = y1 - 10
            for i, text in enumerate(reversed(info_text)):
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(display_frame, 
                            (x1, y_offset - text_size[1] - 5),
                            (x1 + text_size[0] + 10, y_offset + 5),
                            color, -1)
                cv2.putText(display_frame, text, (x1 + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset -= (text_size[1] + 10)
            
            # Draw activity history in corner
            history_y = 30
            # cv2.rectangle(display_frame, (5, 10), (350, history_y + 20 * len(ACTIVITIES) + 30), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Person {person_id} Activity Log:", 
                       (10, history_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            history_y += 25
            
            for act in ACTIVITIES:
                act_data = pdata['activities'][act]
                act_time = act_data['total_time']
                if pdata['current_activity'] == act and act_data['start_time'] is not None:
                    act_time += current_time - act_data['start_time']
                
                if act_time > 0:
                    time_display = format_time(act_time)
                    act_color = ACTIVITY_COLORS.get(act, (255, 255, 255))
                    
                    # Show limit info
                    limit_info = ""
                    act_limit = TIME_LIMITS.get(act)
                    if act_limit is not None:
                        if act_time >= act_limit:
                            limit_info = " [EXCEEDED]"
                            act_color = (0, 0, 255)
                        else:
                            limit_info = f" [Limit: {act_limit}s]"
                    
                    cv2.putText(display_frame, f"{act}: {time_display}{limit_info}", 
                               (10, history_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, act_color, 1)
                    history_y += 20
            
            history_y += 10  # Space between persons
        
        cv2.imshow('YOLO Activity Tracker', display_frame)
    else:
        cv2.imshow('YOLO Activity Tracker', frame)
    
    if cv2.waitKey(1) == 27:  # ESC key
        frame_queue.put(None)  # Stop signal
        break

cap.release()
cv2.destroyAllWindows()

# Print final statistics
print("\n" + "="*50)
print("Final Activity Statistics:")
print("="*50)
for person_id, pdata in tracked_persons.items():
    print(f"\nPerson ID {person_id}:")
    for activity, data in pdata['activities'].items():
        total = data['total_time']
        if pdata['current_activity'] == activity and data['start_time'] is not None:
            total += time.time() - data['start_time']
        if total > 0:
            time_str = format_time(total)
            limit = TIME_LIMITS.get(activity)
            limit_str = f" (Limit: {limit}s)" if limit else ""
            exceeded = " ⚠️ EXCEEDED" if limit and total >= limit else ""
            print(f"  {activity.capitalize()}: {time_str}{limit_str}{exceeded}")