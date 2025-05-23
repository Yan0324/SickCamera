import cv2
import numpy as np
import json
import time
import random
import string
import paho.mqtt.client as mqtt
import threading
from typing import List, Dict, Tuple, Optional
import os
from dataclasses import dataclass
import math

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
class Detection:
    """检测结果数据类，存储目标检测的边界框信息"""
    x: float  # 中心点x坐标
    y: float  # 中心点y坐标
    w: float  # 宽度
    h: float  # 高度
    confidence: float  # 置信度
    point1: Tuple[float, float]  # 左上角点
    point2: Tuple[float, float]  # 右上角点
    point3: Tuple[float, float]  # 右下角点
    point4: Tuple[float, float]  # 左下角点

@dataclass
class BoxInfo:
    """盒子信息数据类，存储检测到的盒子相关信息"""
    is_trapezoid: bool  # 是否为梯形
    detection_count: int  # 检测数量
    top_layer_count: int  # 顶层数量
    layer_height: float  # 层高

@dataclass
class InfoBox:
    """信息盒子数据类，存储检测结果信息"""
    detect_num: int  # 检测数量
    detect_box_average: List[float]  # 检测框平均值列表

class MQTTClient:
    """MQTT客户端类，处理与MQTT服务器的通信"""
    
    def __init__(self):
        """初始化MQTT客户端"""
        self.client = mqtt.Client(CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        """连接回调函数"""
        if rc == 0:
            print("Connected to MQTT broker")
            self.connected = True
            # 订阅相关主题
            self.client.subscribe(DETECTOR_TOPIC, QOS)
            self.client.subscribe(HEIGHT_TOPIC, QOS)
            self.client.subscribe(GRAB_TOPIC, QOS)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        """消息接收回调函数"""
        global CAMERA_RUN, LEFTING_HEIGHT, GRAB
        print(f"Received message on topic {msg.topic}")
        print(f"Message: {msg.payload.decode()}")
        
        # 根据主题设置相应的标志
        if msg.topic == DETECTOR_TOPIC:
            CAMERA_RUN = True
        elif msg.topic == HEIGHT_TOPIC:
            LEFTING_HEIGHT = True
        elif msg.topic == GRAB_TOPIC:
            GRAB = True

    def on_disconnect(self, client, userdata, rc):
        """断开连接回调函数"""
        print(f"Disconnected with result code: {rc}")
        self.connected = False

    def connect(self):
        """连接到MQTT服务器"""
        try:
            self.client.connect(SERVER_ADDRESS, SERVER_PORT)
            self.client.loop_start()
        except Exception as e:
            print(f"Error connecting to MQTT broker: {e}")

    def publish(self, topic: str, payload: str):
        """发布消息到指定主题"""
        if self.connected:
            self.client.publish(topic, payload, QOS)

    def disconnect(self):
        """断开与MQTT服务器的连接"""
        self.client.loop_stop()
        self.client.disconnect()

def generate_random_string(length: int) -> str:
    """生成指定长度的随机字符串"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_random_image_name(extension: str, name_length: int = 10) -> str:
    """生成随机图片文件名"""
    name = generate_random_string(name_length)
    return f"{name}{extension}"

def calculate_square_average(z_values: np.ndarray, width: int, height: int, 
                           center_x: float, center_y: float, radius: int) -> float:
    """计算指定区域内的平均值
    
    Args:
        z_values: Z值数组
        width: 图像宽度
        height: 图像高度
        center_x: 中心点x坐标
        center_y: 中心点y坐标
        radius: 半径
    
    Returns:
        float: 区域内的平均值
    """
    start_x = max(0, int(center_x - radius))
    start_y = max(0, int(center_y - radius))
    end_x = min(width, int(center_x + radius))
    end_y = min(height, int(center_y + radius))

    region = z_values[start_y:end_y, start_x:end_x]
    valid_mask = ~np.isnan(region)
    
    if np.any(valid_mask):
        return np.mean(region[valid_mask])
    return 0.0

def extract_z_values(point_cloud: List[Tuple[float, float, float]], width: int, height: int) -> np.ndarray:
    """从点云数据中提取Z值
    
    Args:
        point_cloud: 点云数据列表
        width: 图像宽度
        height: 图像高度
    
    Returns:
        np.ndarray: Z值数组
    """
    z_values = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if idx < len(point_cloud):
                z_values[i, j] = point_cloud[idx][2]
    return z_values

def convert_to_rect(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    """将中心点和宽高转换为矩形坐标
    
    Args:
        x: 中心点x坐标
        y: 中心点y坐标
        w: 宽度
        h: 高度
    
    Returns:
        Tuple[int, int, int, int]: (left, top, width, height)
    """
    half_w = w / 2.0
    half_h = h / 2.0
    left = int(x - half_w)
    top = int(y - half_h)
    right = int(x + half_w)
    bottom = int(y + half_h)
    return (left, top, right - left, bottom - top)

def calculate_iou(det1: Detection, det2: Detection) -> float:
    """计算两个检测框的IoU（交并比）
    
    Args:
        det1: 第一个检测框
        det2: 第二个检测框
    
    Returns:
        float: IoU值
    """
    rect1 = convert_to_rect(det1.x, det1.y, det1.w, det1.h)
    rect2 = convert_to_rect(det2.x, det2.y, det2.w, det2.h)
    
    # 计算交集
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    rect1_area = rect1[2] * rect1[3]
    rect2_area = rect2[2] * rect2[3]
    union = rect1_area + rect2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def process_detections(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """处理检测结果，去除重叠的检测框
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU阈值
    
    Returns:
        List[Detection]: 处理后的检测结果列表
    """
    i = 0
    while i < len(detections):
        j = i + 1
        while j < len(detections):
            iou = calculate_iou(detections[i], detections[j])
            if iou > iou_threshold:
                if detections[i].confidence < detections[j].confidence:
                    detections.pop(i)
                    i -= 1
                    break
                else:
                    detections.pop(j)
                    j -= 1
            j += 1
        i += 1
    return detections

def draw_detections_obb(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """在图像上绘制检测结果
    
    Args:
        image: 输入图像
        detections: 检测结果列表
    
    Returns:
        np.ndarray: 绘制了检测框的图像
    """
    for det in detections:
        points = np.array([det.point1, det.point2, det.point3, det.point4], np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 2)
        cv2.putText(image, f"{det.confidence:.2f}", 
                   (int(det.x), int(det.y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    """主函数"""
    # 初始化MQTT客户端
    mqtt_client = MQTTClient()
    mqtt_client.connect()

    # 初始化相机（需要实现相机接口）
    camera = None  # 替换为实际的相机初始化

    try:
        while True:
            # 获取相机帧
            frame = None  # 替换为实际的帧捕获
            
            if CAMERA_RUN:
                # 处理帧并运行检测
                detections = []  # 替换为实际的YOLO检测
                
                # 处理检测结果并计算平均值
                averages = []
                for det in detections:
                    average = calculate_square_average(frame, frame.shape[1], frame.shape[0],
                                                     det.x, det.y, 3)
                    averages.append(average)
                
                # 创建InfoBox并转换为JSON
                info_box = InfoBox(len(detections), averages)
                payload = json.dumps({
                    "detectNum": info_box.detect_num,
                    "detectBoxAverage": info_box.detect_box_average
                })
                
                # 发布结果
                mqtt_client.publish("camera/detector_num", payload)
                CAMERA_RUN = False

            elif GRAB:
                # 实现抓取功能
                for _ in range(10):
                    frame = None  # 替换为实际的帧捕获
                    if frame is not None:
                        jpg_name = generate_random_image_name(".jpg")
                        cv2.imwrite(f"./data/{jpg_name}", frame)
                GRAB = False

            elif LEFTING_HEIGHT:
                # 实现高度计算
                height = 0.0  # 替换为实际的高度计算
                mqtt_client.publish("camera/height", str(height))
                LEFTING_HEIGHT = False

            time.sleep(0.1)  # 小延迟以防止CPU过载

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        mqtt_client.disconnect()
        if camera is not None:
            camera.release()

if __name__ == "__main__":
    main() 