#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import cv2
import numpy as np
from vision import QtVisionSick
from common.PointCloud.PointCloud import convertToPointCloud, writePointCloudToFile
from common.data_io.DepthToImage import saveDepthToPng
from rknn_yolo import RKNN_YOLO

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SickPointCloudGenerator:
    def __init__(self, ip_addr="192.168.10.5", port=2122, protocol="Cola2", model_path="obb_exported.float.rknn"):
        """
        初始化点云生成器
        
        Args:
            ip_addr (str): 相机IP地址
            port (int): 相机控制端口
            protocol (str): 通信协议
            model_path (str): RKNN模型路径
        """
        self.camera = QtVisionSick(ipAddr=ip_addr, port=port, protocol=protocol)
        self.output_dir = 'point_cloud_output'
        # 初始化目标检测模型
        try:
            self.detector = RKNN_YOLO(model_path)
            logger.info("目标检测模型加载成功")
        except Exception as e:
            logger.error(f"目标检测模型加载失败: {str(e)}")
            self.detector = None
        
        # 创建输出目录
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"无法创建输出目录: {str(e)}")
            sys.exit(1)
            
    def connect_camera(self):
        """连接相机"""
        logger.info("正在连接相机...")
        if self.camera.connect():
            logger.info("相机连接成功")
            return True
        else:
            logger.error("相机连接失败")
            return False
            
    def generate_point_cloud(self, frame_count=1, detect_objects=True):
        """
        生成点云数据
        
        Args:
            frame_count (int): 要处理的帧数
            detect_objects (bool): 是否进行目标检测
        """
        if not self.camera.is_connected:
            logger.error("相机未连接")
            return False
        print(self.camera.deviceControl.getStatus())
        self.camera.deviceControl.singleStep()
        try:
            for i in range(frame_count):
                logger.info(f"正在处理第 {i+1} 帧...")
                
                # 获取深度和强度数据
                success, depth_data, intensity_image = self.camera.get_frame_no_step()
                if not success:
                    logger.error(f"获取第 {i+1} 帧数据失败")
                    continue
                    
                # 获取相机参数
                camera_params = self.camera.camera_params
                if camera_params is None:
                    logger.error("无法获取相机参数")
                    continue
                
                # 进行目标检测
                detection_results = None
                if detect_objects and self.detector is not None:
                    # 将强度图转换为BGR格式用于检测
                    intensity_bgr = cv2.cvtColor(intensity_image, cv2.COLOR_GRAY2BGR)
                    detection_results = self.detector.detect(intensity_bgr)
                    
                    # 在强度图上绘制检测结果
                    if detection_results:
                        intensity_bgr = self.detector.draw_result(intensity_bgr, detection_results)
                        # 保存检测结果图像
                        detection_img_path = os.path.join(self.output_dir, f"detection_{i+1}.png")
                        cv2.imwrite(detection_img_path, intensity_bgr)
                        logger.info(f"检测结果已保存到: {detection_img_path}")
                    
                # 转换为点云数据
                world_coordinates, dist_data = convertToPointCloud(
                    depth_data,
                    intensity_image.flatten().tolist(),
                    [255] * len(depth_data),  # 使用默认置信度
                    camera_params,
                    False  # 非立体相机
                )
                
                # 保存点云数据
                ply_filename = os.path.join(self.output_dir, f"point_cloud_{i+1}.ply")
                writePointCloudToFile(ply_filename, world_coordinates)
                logger.info(f"点云数据已保存到: {ply_filename}")
                
                # 保存深度图和强度图
                saveDepthToPng(
                    self.output_dir,
                    dist_data,
                    intensity_image.flatten().tolist(),
                    [255] * len(depth_data),
                    camera_params,
                    i,
                    False
                )
                logger.info(f"深度图和强度图已保存到: {self.output_dir}")
                
                # 如果有检测结果，保存检测框的3D坐标
                if detection_results:
                    detection_3d_path = os.path.join(self.output_dir, f"detection_3d_{i+1}.txt")
                    with open(detection_3d_path, 'w') as f:
                        for box in detection_results:
                            # 获取检测框中心点的深度值
                            center_x = int((box.pt1x + box.pt3x) / 2)
                            center_y = int((box.pt1y + box.pt3y) / 2)
                            if 0 <= center_x < len(depth_data) and 0 <= center_y < len(depth_data[0]):
                                depth = depth_data[center_y][center_x]
                                f.write(f"Class: {box.classId}, Score: {box.score:.2f}, "
                                       f"Center: ({center_x}, {center_y}), Depth: {depth}\n")
                    logger.info(f"3D检测结果已保存到: {detection_3d_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"生成点云数据时出错: {str(e)}")
            return False
            
    def disconnect_camera(self):
        """断开相机连接"""
        self.camera.disconnect()
        if self.detector:
            self.detector.release()
        logger.info("相机已断开连接")
        
    def __enter__(self):
        """上下文管理器入口"""
        self.connect_camera()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect_camera()

def main():
    # 使用示例
    with SickPointCloudGenerator() as generator:
        # 生成一帧点云数据，并进行目标检测
        generator.generate_point_cloud(frame_count=1, detect_objects=True)

if __name__ == "__main__":
    main() 