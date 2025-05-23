#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import os
import logging
from SickSDK import QtVisionSick
from rknn_yolo import RKNN_YOLO

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 将日志级别改为WARNING，只显示警告和错误
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 设置其他模块的日志级别
logging.getLogger('common.Stream').setLevel(logging.WARNING)
logging.getLogger('common.Streaming').setLevel(logging.WARNING)
logging.getLogger('common.Control').setLevel(logging.WARNING)

class BoxCounter:
    def __init__(self, ip_address="192.168.10.5", port=2122, protocol="Cola2"):
        self.camera = QtVisionSick(ip_address, port, protocol)
        
        # 创建输出目录
        self.output_dir = 'detection_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化检测模型
        model_path = 'obb_exported.float.rknn'
        self.detector = RKNN_YOLO(model_path)
        self.logger.info("检测模型加载完成")

    def process_frame(self, intensity_image, depth_data):
        """
        处理单帧数据
        Args:
            intensity_image: 强度图像
            depth_data: 深度数据
        Returns:
            vis_image: 可视化图像
            boxes: 检测到的箱子列表
        """
        try:
            # 创建可视化图像
            # 首先将强度图像转换为8位无符号整数类型
            intensity_image = np.clip(intensity_image, 0, 255).astype(np.uint8)
            
            # 如果是单通道图像，转换为3通道
            if len(intensity_image.shape) == 2:
                vis_image = cv2.cvtColor(intensity_image, cv2.COLOR_GRAY2BGR)
            else:
                vis_image = intensity_image.copy()
            
            # 进行箱子检测
            boxes = self.detector.detect(vis_image)
            print(boxes)
            
            # 在图像上绘制检测结果
            if boxes:
                vis_image = self.detector.draw_result(vis_image, boxes)
                # 显示检测到的箱子数量
                cv2.putText(vis_image, f"Boxes: {len(boxes)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return vis_image, boxes
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return None, None

    def close(self):
        """关闭相机连接和释放模型资源"""
        self.camera.disconnect()
        self.detector.release()

def main():
    # 创建计数器实例
    counter = BoxCounter()
    try:
        # 连接相机，使用连续模式
        if not counter.camera.connect(use_single_step=False):
            print("无法连接到相机，程序退出")
            return
            
            
        print("开始实时检测...")
        while True:
            try:
                # 获取一帧数据
                success, depth_data, intensity_image = counter.camera.get_frame()
                if not success:
                    continue
                    
                # 处理数据
                vis_image, boxes = counter.process_frame(intensity_image, depth_data)
                if vis_image is not None:
                    # 显示处理结果
                    cv2.imshow('Detection Result', vis_image)
                        
                    if depth_data is not None:
                        # 将深度数据转换为图像格式
                        depth_image = np.array(depth_data).reshape((counter.camera.camera_params.height, 
                                                                 counter.camera.camera_params.width))
                        # 归一化深度图以便显示
                        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imshow('Depth Image', depth_vis)
                        
                    # 按q键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"处理帧时出错: {str(e)}")
                time.sleep(0.1)  # 添加短暂延迟避免CPU占用过高
                continue
                    
    except KeyboardInterrupt:
        print("\n程序终止")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
    finally:
        counter.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 