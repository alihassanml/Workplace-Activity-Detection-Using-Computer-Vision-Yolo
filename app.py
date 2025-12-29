from ultralytics import YOLO  # yolo library
import cv2 # computer visiion
from playsound import playsound 
import threading
import os
model = YOLO('models/yolov8s.pt') # Model load  1

print("Model classes:", model.names)


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break  

    results = model(frame)  # result frame to model 
    result = results[0]

    for box in result.boxes:
        cls = int(box.cls)
        class_name = model.names[cls]

    annotated_frame = result.plot() 
    cv2.imshow('YOLO Inference', annotated_frame)
    
    if cv2.waitKey(1) == 27:  # 
        break

cap.release()
cv2.destroyAllWindows()
