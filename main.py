import cv2
import cvzone
import math
import requests
from ultralytics import YOLO
from datetime import datetime

cap = cv2.VideoCapture('fall.mp4')
model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

fall_detected = False

# Assuming you have a reference ID and value that you want to send with the image
ref = "your_ref_id"  # Replace with actual reference ID
value = True  # Example value, change according to your logic

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (980, 740))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                if threshold < 0 and not fall_detected:
                    now = datetime.now()
                    frame_filename = now.strftime("%d-%m-%Y-%H-%M") + '.jpg'
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    print("log: Fall Detected")
                    cv2.imwrite(frame_filename, frame)
                    fall_detected = True

                    # Prepare the data to send in the POST request
                    with open(frame_filename, 'rb') as f:
                        files = {'image': (frame_filename, f, 'image/jpeg')}
                        # Add the ref and value fields to the data
    
                        response = requests.post('https://iot-fall-be.vercel.app/realtime/write/fall_down', files=files)

                    if response.status_code == 200:
                        print("Frame uploaded successfully:", response.json())
                    else:
                        print("Failed to upload frame:", response.status_code, response.text)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
