import cv2
import cvzone
import math
import requests
import socket
import pickle
import struct
import os
from ultralytics import YOLO
from datetime import datetime

# Server setup
host = '192.168.1.10'  # Listen on all available interfaces
port = 12345      # Same port as the client

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print("Waiting for a connection...")
client_socket, addr = server_socket.accept()
print(f"Connection from {addr} has been established.")

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

fall_detected = False
ref = "your_ref_id"  # Replace with actual reference ID
value = True  # Example value, change according to your logic

# Ensure the Images directory exists
os.makedirs('Images', exist_ok=True)

try:
    data_buffer = b""
    payload_size = struct.calcsize("Q")
    
    while True:
        # Retrieve message size and frame data from the client
        while len(data_buffer) < payload_size:
            packet = client_socket.recv(4096)
            if not packet:  # Connection closed
                break
            data_buffer += packet
        
        if len(data_buffer) < payload_size:
            break

        packed_msg_size = data_buffer[:payload_size]
        data_buffer = data_buffer[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data_buffer) < msg_size:
            data_buffer += client_socket.recv(4096)
        
        frame_data = data_buffer[:msg_size]
        data_buffer = data_buffer[msg_size:]
        
        # Deserialize the frame
        frame = pickle.loads(frame_data)

        # Process the frame
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
                        # Save the image in the Images folder with timestamp
                        frame_filename = os.path.join('images', now.strftime("%d-%m-%Y-%H-%M") + '.jpg')
                        cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                        print("log: Fall Detected")
                        cv2.imwrite(frame_filename, frame)
                        fall_detected = True

                        # Prepare the data to send in the POST request
                        with open(frame_filename, 'rb') as f:
                            files = {'image': (frame_filename, f, 'image/jpeg')}
                            response = requests.post('https://iot-fall-be.vercel.app/realtime/write/fall_down', files=files)

                        if response.status_code == 200:
                            print("Frame uploaded successfully:", response.json())
                        else:
                            print("Failed to upload frame:", response.status_code, response.text)

        # Display the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Server closed.")
