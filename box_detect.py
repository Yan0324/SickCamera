#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from common.Streaming import Data
from common.PointCloud.PointCloud import convertToPointCloud
import time
import os
import logging
from common.Control import Control
from common.Stream import Streaming
from common.Streaming.BlobServerConfiguration import BlobClientConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SickCamera:
    def __init__(self, ip_address="192.168.10.5", port=2122, protocol="Cola2", streaming_port=2114):
        self.ip_address = ip_address
        self.port = port
        self.protocol = protocol
        self.streaming_port = streaming_port
        self.logger = logging.getLogger(__name__)
        
        # 初始化控制连接
        self.device_control = Control(ip_address, protocol, port)
        self.streaming_device = None
        self.is_connected = False
        
    def connect(self):
        """连接相机并配置数据流"""
        try:
            # 打开控制连接
            self.device_control.open()
            
            # 登录设备
            self.device_control.login(Control.USERLEVEL_SERVICE, 'CUST_SERV')
            
            # 获取设备信息
            name, version = self.device_control.getIdent()
            self.logger.info(f"Connected to device: {name.decode('utf-8')}, version: {version.decode('utf-8')}")
            
            # 配置数据流
            streaming_settings = BlobClientConfig()
            
            # 设置TCP协议和端口
            streaming_settings.setTransportProtocol(self.device_control, streaming_settings.PROTOCOL_TCP)
            streaming_settings.setBlobTcpPort(self.device_control, self.streaming_port)
            
            # 启动数据流
            self.streaming_device = Streaming(self.ip_address, self.streaming_port)
            self.streaming_device.openStream()
            
            # 登出控制连接
            self.device_control.logout()
            
            # 启动数据流
            self.device_control.startStream()
            
            self.is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to camera: {str(e)}")
            self.is_connected = False
            return False
            
    def get_frame(self):
        """获取一帧数据"""
        if not self.is_connected:
            self.logger.error("Camera not connected")
            return None, None, None
            
        try:
            # 获取数据帧
            self.streaming_device.getFrame()
            whole_frame = self.streaming_device.frame
            
            # 解析数据
            frame_data = Data.Data()
            frame_data.read(whole_frame)
            
            if frame_data.hasDepthMap:
                # 获取深度数据和强度数据
                depth_data = list(frame_data.depthmap.distance)
                intensity_data = list(frame_data.depthmap.intensity)
                confidence_data = list(frame_data.depthmap.confidence)
                
                # 重塑数据为图像格式
                num_rows = frame_data.cameraParams.height
                num_cols = frame_data.cameraParams.width
                
                depth_image = np.array(depth_data).reshape((num_rows, num_cols))
                intensity_image = np.array(intensity_data).reshape((num_rows, num_cols))
                
                return intensity_image, depth_image, frame_data
            else:
                self.logger.warning("No depth map data in frame")
                return None, None, None
                
        except Exception as e:
            self.logger.error(f"Error getting frame: {str(e)}")
            return None, None, None
            
    def disconnect(self):
        """断开相机连接"""
        try:
            if self.streaming_device:
                self.streaming_device.closeStream()
            
            if self.device_control:
                self.device_control.login(Control.USERLEVEL_AUTH_CLIENT, '123456')
                self.device_control.logout()
                
            self.is_connected = False
            self.logger.info("Camera disconnected")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting camera: {str(e)}")

class VolumeAnalyzer:
    def __init__(self):
        self.voxel_size = 0.005  # 体素大小（米）
        self.distance_threshold = 0.5  # 距离阈值（米）
        self.std_dev_multiplier = 1.0  # 标准差乘数，用于统计滤波
        
    def filter_noise(self, points):
        """
        使用多种方法过滤点云噪声
        Args:
            points: 点云数据 [x, y, z, ...] (单位：米)
        Returns:
            filtered_points: 过滤后的点云数据
        """
        if len(points) == 0:
            return points
            
        # 1. 距离滤波：移除距离异常的点
        distances = np.linalg.norm(points[:, :3], axis=1)
        distance_mask = distances < self.distance_threshold
        points = points[distance_mask]
        
        if len(points) == 0:
            return points
            
        # 2. 统计滤波：移除离群点
        mean = np.mean(points[:, :3], axis=0)
        std = np.std(points[:, :3], axis=0)
        distance = np.abs(points[:, :3] - mean)
        mask = np.all(distance < self.std_dev_multiplier * std, axis=1)
        points = points[mask]
        
        return points
        
    def calculate_volume(self, points):
        """
        使用体素化方法计算点云数据的总体积
        Args:
            points: 点云数据 [x, y, z, ...] (单位：毫米)
        Returns:
            total_volume: 总体积（立方米）
        """
        if len(points) == 0:
            return 0
            
        # 1. 提取3D坐标并转换为米
        points_3d = np.array([p[:3] for p in points]) / 1000.0
        
        # 2. 过滤无效点
        valid_points = points_3d[points_3d[:, 2] > 0]
        
        if len(valid_points) == 0:
            return 0
            
        # 3. 应用噪声过滤
        filtered_points = self.filter_noise(valid_points)
        
        if len(filtered_points) == 0:
            return 0
            
        # 4. 计算点云的边界框
        min_coords = np.min(filtered_points, axis=0)
        max_coords = np.max(filtered_points, axis=0)
        
        # 5. 创建体素网格
        voxel_grid = np.zeros(
            (
                int((max_coords[0] - min_coords[0]) / self.voxel_size) + 1,
                int((max_coords[1] - min_coords[1]) / self.voxel_size) + 1,
                int((max_coords[2] - min_coords[2]) / self.voxel_size) + 1
            ),
            dtype=bool
        )
        
        # 6. 将点云数据体素化
        for point in filtered_points:
            x = int((point[0] - min_coords[0]) / self.voxel_size)
            y = int((point[1] - min_coords[1]) / self.voxel_size)
            z = int((point[2] - min_coords[2]) / self.voxel_size)
            if 0 <= x < voxel_grid.shape[0] and 0 <= y < voxel_grid.shape[1] and 0 <= z < voxel_grid.shape[2]:
                voxel_grid[x, y, z] = True
        
        # 7. 计算体积（体素数量 * 体素体积）
        total_volume = np.sum(voxel_grid) * (self.voxel_size ** 3)
            
        return total_volume

class BoxCounter:
    def __init__(self, ip_address="192.168.10.5", port=2122, protocol="Cola2", streaming_port=2114):
        self.camera = SickCamera(ip_address, port, protocol, streaming_port)
        self.volume_analyzer = VolumeAnalyzer()
        
        # 创建输出目录
        self.output_dir = 'detection_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)

    def process_frame(self, intensity_image, depth_image, frame_data):
        """
        处理单帧数据
        Args:
            intensity_image: 强度图像
            depth_image: 深度图像
            frame_data: 相机数据帧
        Returns:
            total_volume: 总体积
            vis_image: 可视化图像
        """
        if frame_data is None or not frame_data.hasDepthMap:
            self.logger.error("Invalid frame data")
            return None, None
            
        try:
            # 确保数据格式正确
            depth_data = np.array(frame_data.depthmap.distance)
            intensity_data = np.array(frame_data.depthmap.intensity)
            confidence_data = np.array(frame_data.depthmap.confidence)
            
            # 获取点云数据
            world_coordinates, dist_data = convertToPointCloud(
                depth_data,
                intensity_data,
                confidence_data,
                frame_data.cameraParams,
                frame_data.xmlParser.stereo
            )
            
            # 确保world_coordinates是numpy数组
            if isinstance(world_coordinates, list):
                world_coordinates = np.array(world_coordinates)
            
            # 检查数据维度并正确处理
            if len(world_coordinates.shape) == 3:
                # 如果是3D数组，重塑为2D
                points = world_coordinates.reshape(-1, world_coordinates.shape[2])
            else:
                # 如果已经是2D数组，直接使用
                points = world_coordinates
            
            # 计算体积
            total_volume = self.volume_analyzer.calculate_volume(points)
            
            # 创建可视化图像
            # 首先将强度图像转换为8位无符号整数类型
            intensity_image = np.clip(intensity_image, 0, 255).astype(np.uint8)
            
            # 如果是单通道图像，转换为3通道
            if len(intensity_image.shape) == 2:
                vis_image = cv2.cvtColor(intensity_image, cv2.COLOR_GRAY2BGR)
            else:
                vis_image = intensity_image.copy()
            
            # 在图像上显示体积信息
            cv2.putText(vis_image, f"Volume: {total_volume:.3f}m³", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 添加调试信息
            cv2.putText(vis_image, f"Points: {len(points)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 打印点云数据的范围
            if len(points) > 0:
                points_3d = np.array([p[:3] for p in points])
                min_coords = np.min(points_3d, axis=0)
                max_coords = np.max(points_3d, axis=0)
                print(f"\n点云范围:")
                print(f"X: {min_coords[0]:.2f} to {max_coords[0]:.2f} mm")
                print(f"Y: {min_coords[1]:.2f} to {max_coords[1]:.2f} mm")
                print(f"Z: {min_coords[2]:.2f} to {max_coords[2]:.2f} mm")
            
            return total_volume, vis_image
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return None, None

    def close(self):
        """关闭相机连接"""
        self.camera.disconnect()

def main():
    # 创建计数器实例
    counter = BoxCounter()
    try:
        # 连接相机
        if not counter.camera.connect():
            print("无法连接到相机，程序退出")
            return
            
        print("开始实时检测...")
        while True:
            # 获取一帧数据
            intensity_image, depth_image, frame_data = counter.camera.get_frame()
            if frame_data is None:
                continue
                
            # 处理数据
            total_volume, vis_image = counter.process_frame(intensity_image, depth_image, frame_data)
            if total_volume is not None:
                print(f"\r检测到的体积: {total_volume:.5f}m³", end="")
                
                # 显示处理结果
                if vis_image is not None:
                    cv2.imshow('Detection Result', vis_image)
                    
                if depth_image is not None:
                    # 归一化深度图以便显示
                    depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow('Depth Image', depth_vis)
                    
                # 按q键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\n程序终止")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    finally:
        counter.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 