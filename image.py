from ultralytics import YOLO
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model = YOLO('models/yolov8s.pt')
url  = 'images/images.jpeg'
results = model(url)

result = results[0]

image_with_detections = result.plot()

cv2.imshow('Detections', image_with_detections)

cv2.waitKey(0)
cv2.destroyAllWindows()
