import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import time
import random
import string
import os
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from common.VisionaryControl import VisionaryControl
from common.VisionaryDataStream import VisionaryDataStream
from common.VisionaryTMiniData import VisionaryTMiniData
from common.PointXYZ import PointXYZ
from utils.logging import setup_logger
from draw.cv_draw import draw_detections_obb
from task.yolov8_custom import Yolov8Custom

# MQTT Configuration
SERVER_ADDRESS = "192.168.0.164"
SERVER_PORT = 1883
CLIENT_ID = "rk3588"
DETECTOR_TOPIC = "camera/start_detector"
HEIGHT_TOPIC = "camera/lifting_height"
GRAB_TOPIC = "warehouse/start_grab"
QOS = 2
N_RETRY_ATTEMPTS = 5

# Global flags
CAMERA_RUN = False
LEFTING_HEIGHT = False
GRAB = False

@dataclass
class BoxInfo:
    is_trapezoid: bool
    detection_count: int
    top_layer_count: int
    layer_height: int

@dataclass
class InfoBox:
    detect_num: int
    detect_box_average: List[float]

class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client(CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print("Connected to MQTT broker")
            self.client.subscribe(DETECTOR_TOPIC, QOS)
            self.client.subscribe(HEIGHT_TOPIC, QOS)
            self.client.subscribe(GRAB_TOPIC, QOS)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        global CAMERA_RUN, LEFTING_HEIGHT, GRAB
        print(f"Received message on topic {msg.topic}")
        if msg.topic == DETECTOR_TOPIC:
            CAMERA_RUN = True
        elif msg.topic == HEIGHT_TOPIC:
            LEFTING_HEIGHT = True
        elif msg.topic == GRAB_TOPIC:
            GRAB = True

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        print(f"Disconnected with result code: {rc}")

    def connect(self):
        try:
            self.client.connect(SERVER_ADDRESS, SERVER_PORT)
            self.client.loop_start()
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")

    def publish(self, topic: str, payload: str):
        if self.connected:
            self.client.publish(topic, payload, QOS)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

class VisionaryDetector:
    def __init__(self, device_ip: str = "192.168.10.5", device_port: int = 2114):
        self.device_ip = device_ip
        self.device_port = device_port
        self.data_handler = VisionaryTMiniData()
        self.data_stream = VisionaryDataStream(self.data_handler)
        self.visionary_control = VisionaryControl()
        self.mqtt_client = MQTTClient()
        self.yolo = None

    def connect_to_camera(self) -> bool:
        max_retries = 10
        retry_interval = 5  # seconds

        for retry in range(max_retries):
            try:
                if not self.data_stream.open(self.device_ip, self.device_port):
                    print(f"Failed to connect to camera. Retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                    continue
                
                if not self.visionary_control.open(VisionaryControl.ProtocolType.COLA_2, 
                                                 self.device_ip, 5000):
                    print("Failed to open control connection to device.")
                    return False

                print(f"Device Ident: {self.visionary_control.get_device_ident()}")
                self.visionary_control.start_acquisition()
                return True
            except Exception as e:
                print(f"Error connecting to camera: {e}")
                time.sleep(retry_interval)

        print(f"Failed to connect to camera after {max_retries} attempts")
        return False

    def load_model(self, model_path: str):
        self.yolo = Yolov8Custom()
        self.yolo.load_model(model_path)
        self.yolo.set_static_params(0.1, 0.8, "top_1_labels_list.txt", 1)

    def calculate_square_average(self, z_values: List[List[float]], width: int, height: int,
                               center_x: float, center_y: float, radius: int) -> float:
        count = 0
        sum_val = 0.0

        start_x = max(0, int(center_x - radius))
        start_y = max(0, int(center_y - radius))
        end_x = min(width, int(center_x + radius))
        end_y = min(height, int(center_y + radius))

        for i in range(start_y, end_y):
            for j in range(start_x, end_x):
                if not np.isnan(z_values[i][j]):
                    sum_val += z_values[i][j]
                    count += 1

        return sum_val / count if count > 0 else 0.0

    def extract_z_values(self, point_cloud: List[PointXYZ], width: int, height: int) -> List[List[float]]:
        z_values = [[0.0 for _ in range(width)] for _ in range(height)]
        for i in range(height):
            for j in range(width):
                index = i * width + j
                if index < len(point_cloud):
                    z_values[i][j] = point_cloud[index].z
        return z_values

    def process_frame(self):
        if not self.data_stream.get_next_frame():
            return

        if CAMERA_RUN:
            width = self.data_handler.get_camera_parameters().width
            height = self.data_handler.get_camera_parameters().height
            
            # Get intensity map and point cloud
            intensity_map = self.data_handler.get_intensity_map()
            point_cloud = []
            self.data_handler.generate_point_cloud(point_cloud)
            self.data_handler.transform_point_cloud(point_cloud)

            # Process intensity image
            image = np.array(intensity_map, dtype=np.uint16).reshape(height, width)
            normalized_map = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            adjusted_image = cv2.convertScaleAbs(image, alpha=0.3, beta=10)
            cv2.imwrite("adjusted_image.jpg", adjusted_image)
            img = cv2.imread("adjusted_image.jpg")

            # Run YOLO detection
            objects = self.yolo.run_obb(img)
            
            # Calculate averages for detected objects
            z_values = self.extract_z_values(point_cloud, width, height)
            averages = []
            for obj in objects:
                average = self.calculate_square_average(z_values, width, height, 
                                                     obj.x, obj.y, 3)
                averages.append(average)

            # Create and send info box
            info_box = InfoBox(len(objects), averages)
            payload = json.dumps({
                "detectNum": info_box.detect_num,
                "detectBoxAverage": info_box.detect_box_average
            })
            
            # Draw detections and save result
            draw_detections_obb(img, objects)
            cv2.imwrite("result_from_read.jpg", img)
            
            # Publish results
            self.mqtt_client.publish("camera/detector_num", payload)
            global CAMERA_RUN
            CAMERA_RUN = False

    def run(self):
        self.mqtt_client.connect()
        
        try:
            while True:
                self.process_frame()
                time.sleep(0.1)  # Small delay to prevent CPU overload
        except KeyboardInterrupt:
            print("Stopping detector...")
        finally:
            self.mqtt_client.disconnect()
            self.visionary_control.close()
            self.data_stream.close()

def main():
    detector = VisionaryDetector()
    if not detector.connect_to_camera():
        print("Failed to connect to camera")
        return

    model_path = "obb_exported.float.rknn"  # Update with your model path
    detector.load_model(model_path)
    detector.run()

if __name__ == "__main__":
    main() 