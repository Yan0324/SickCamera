"""
@Description :   Define the model loading class, to export rknn model and using.
                 the file must run on linux and install rknn-toolkit2 with python.
                 more information refer to https://github.com/airockchip/rknn-toolkit2/tree/master
@Author      :   Cao Yingjie
@Time        :   2025/04/23 08:47:48
"""

import os
import sys
import urllib
import urllib.request
import time
import numpy as np
import argparse
import cv2,math
from math import ceil
from itertools import product as product
from shapely.geometry import Polygon

from rknn.api import RKNN

class RKNN_YOLO:
    """
    RKNN YOLO模型封装类
    用于加载和运行RKNN模型进行目标检测
    """
    
    def __init__(self, model_path, target='rk3588', device_id=None):
        """
        初始化RKNN YOLO模型
        
        Args:
            model_path (str): RKNN模型路径
            target (str, optional): 目标RKNPU平台. 默认为 'rk3588'
            device_id (str, optional): 设备ID. 默认为 None
        """
        self.CLASSES = ['box']
        self.meshgrid = []
        self.class_num = len(self.CLASSES)
        self.head_num = 3
        self.strides = [8, 16, 32]
        self.map_size = [[80, 80], [40, 40], [20, 20]]
        self.reg_num = 16
        self.input_height = 640
        self.input_width = 640
        self.nms_thresh = 0.7
        self.object_thresh = 0.7
        
        # 初始化RKNN
        self.rknn = RKNN(verbose=True)
        
        # 加载模型
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f'Load RKNN model "{model_path}" failed!')
            
        # 初始化运行时环境
        ret = self.rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            raise RuntimeError('Init runtime environment failed!')
            
        # 生成网格
        self._generate_meshgrid()
        
    def _generate_meshgrid(self):
        """生成网格坐标"""
        for index in range(self.head_num):
            for i in range(self.map_size[index][0]):
                for j in range(self.map_size[index][1]):
                    self.meshgrid.append(j + 0.5)
                    self.meshgrid.append(i + 0.5)
                    
    def _get_covariance_matrix(self, boxes):
        """计算协方差矩阵"""
        a, b, c = boxes.w, boxes.h, boxes.angle
        cos = math.cos(c)
        sin = math.sin(c)
        cos2 = math.pow(cos, 2)
        sin2 = math.pow(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
        
    def _probiou(self, obb1, obb2, eps=1e-7):
        """计算旋转框IOU"""
        x1, y1 = obb1.x, obb1.y
        x2, y2 = obb2.x, obb2.y
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = self._get_covariance_matrix(obb2)

        t1 = (((a1 + a2) * math.pow((y1 - y2), 2) + (b1 + b2) * math.pow((x1 - x2), 2)) / ((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2) + eps)) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2) + eps)) * 0.5

        temp1 = (a1 * b1 - math.pow(c1, 2)) if (a1 * b1 - math.pow(c1, 2)) > 0 else 0
        temp2 = (a2 * b2 - math.pow(c2, 2)) if (a2 * b2 - math.pow(c2, 2)) > 0 else 0
        t3 = math.log((((a1 + a2) * (b1 + b2) - math.pow((c1 + c2), 2)) / (4 * math.sqrt((temp1 * temp2)) + eps)+ eps)) * 0.5

        if (t1 + t2 + t3) > 100:
            bd = 100
        elif (t1 + t2 + t3) < eps:
            bd = eps
        else:
            bd = t1 + t2 + t3
        hd = math.sqrt((1.0 - math.exp(-bd) + eps))
        return 1 - hd
        
    def _nms_rotated(self, boxes, nms_thresh):
        """旋转框NMS"""
        pred_boxes = []
        sort_boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
        for i in range(len(sort_boxes)):
            if sort_boxes[i].classId != -1:
                pred_boxes.append(sort_boxes[i])
                for j in range(i + 1, len(sort_boxes), 1):
                    ious = self._probiou(sort_boxes[i], sort_boxes[j])
                    if ious > nms_thresh:
                        sort_boxes[j].classId = -1
        return pred_boxes
        
    def _sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + math.exp(-x))
        
    def _xywhr2xyxyxyxy(self, x, y, w, h, angle):
        """将中心点坐标转换为四个角点坐标"""
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)

        vec1x= w / 2 * cos_value
        vec1y = w / 2 * sin_value
        vec2x = -h / 2 * sin_value
        vec2y = h / 2 * cos_value

        pt1x = x + vec1x + vec2x
        pt1y = y + vec1y + vec2y

        pt2x = x + vec1x - vec2x
        pt2y = y + vec1y - vec2y

        pt3x = x - vec1x - vec2x
        pt3y = y - vec1y - vec2y

        pt4x = x - vec1x + vec2x
        pt4y = y - vec1y + vec2y
        return pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y
        
    def _postprocess(self, out):
        """后处理函数"""
        detect_result = []
        output = []
        for i in range(len(out)):
            output.append(out[i].reshape((-1)))

        gridIndex = -2
        cls_index = 0
        cls_max = 0

        for index in range(self.head_num):
            reg = output[index * 2 + 0]
            cls = output[index * 2 + 1]
            ang = output[self.head_num * 2 + index]

            for h in range(self.map_size[index][0]):
                for w in range(self.map_size[index][1]):
                    gridIndex += 2

                    if 1 == self.class_num:
                        cls_max = self._sigmoid(cls[0 * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w])
                        cls_index = 0
                    else:
                        for cl in range(self.class_num):
                            cls_val = cls[cl * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w]
                            if 0 == cl:
                                cls_max = cls_val
                                cls_index = cl
                            else:
                                if cls_val > cls_max:
                                    cls_max = cls_val
                                    cls_index = cl
                        cls_max = self._sigmoid(cls_max)

                    if cls_max > self.object_thresh:
                        regdfl = []
                        for lc in range(4):
                            sfsum = 0
                            locval = 0
                            for df in range(self.reg_num):
                                temp = math.exp(reg[((lc * self.reg_num) + df) * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w])
                                reg[((lc * self.reg_num) + df) * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w] = temp
                                sfsum += temp

                            for df in range(self.reg_num):
                                sfval = reg[((lc * self.reg_num) + df) * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w] / sfsum
                                locval += sfval * df
                            regdfl.append(locval)

                        angle = (self._sigmoid(ang[h * self.map_size[index][1] + w]) - 0.25) * math.pi

                        left, top, right, bottom = regdfl[0], regdfl[1], regdfl[2], regdfl[3]
                        cos, sin = math.cos(angle), math.sin(angle)
                        fx = (right - left) / 2
                        fy = (bottom - top) / 2

                        cx = ((fx * cos - fy * sin) + self.meshgrid[gridIndex + 0]) * self.strides[index]
                        cy = ((fx * sin + fy * cos) + self.meshgrid[gridIndex + 1])* self.strides[index]
                        cw = (left + right) * self.strides[index]
                        ch = (top + bottom) * self.strides[index]

                        box = CSXYWHR(cls_index, cls_max, cx, cy, cw, ch, angle)
                        detect_result.append(box)

        pred_boxes = self._nms_rotated(detect_result, self.nms_thresh)
        result = []
        
        for i in range(len(pred_boxes)):
            classid = pred_boxes[i].classId
            score = pred_boxes[i].score
            cx = pred_boxes[i].x
            cy = pred_boxes[i].y
            cw = pred_boxes[i].w
            ch = pred_boxes[i].h
            angle = pred_boxes[i].angle

            bw_ = cw if cw > ch else ch
            bh_ = ch if cw > ch else cw
            bt = angle % math.pi if cw > ch else (angle + math.pi / 2) % math.pi

            pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y = self._xywhr2xyxyxyxy(cx, cy, bw_, bh_, bt)
            bbox = DetectBox(classid, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y, angle)
            result.append(bbox)
            
        return result
        
    def detect(self, image):
        """
        对输入图像进行目标检测
        
        Args:
            image (numpy.ndarray): 输入图像，BGR格式
            
        Returns:
            list: 检测结果列表，每个元素为DetectBox对象
        """
        # 预处理
        image_h, image_w = image.shape[:2]
        image = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, 0)
        
        # 推理
        results = self.rknn.inference(inputs=[image])
        
        # 后处理
        pred_boxes = self._postprocess(results)
        
        # 转换回原始图像尺寸
        for box in pred_boxes:
            box.pt1x = int(box.pt1x / self.input_width * image_w)
            box.pt1y = int(box.pt1y / self.input_height * image_h)
            box.pt2x = int(box.pt2x / self.input_width * image_w)
            box.pt2y = int(box.pt2y / self.input_height * image_h)
            box.pt3x = int(box.pt3x / self.input_width * image_w)
            box.pt3y = int(box.pt3y / self.input_height * image_h)
            box.pt4x = int(box.pt4x / self.input_width * image_w)
            box.pt4y = int(box.pt4y / self.input_height * image_h)
            
        return pred_boxes
        
    def draw_result(self, image, boxes):
        """
        在图像上绘制检测结果
        
        Args:
            image (numpy.ndarray): 输入图像
            boxes (list): 检测结果列表
            
        Returns:
            numpy.ndarray: 绘制了检测框的图像
        """
        for box in boxes:
            # 绘制检测框
            cv2.line(image, (box.pt1x, box.pt1y), (box.pt2x, box.pt2y), (255, 0, 0), 2)
            cv2.line(image, (box.pt2x, box.pt2y), (box.pt3x, box.pt3y), (255, 0, 0), 2)
            cv2.line(image, (box.pt3x, box.pt3y), (box.pt4x, box.pt4y), (255, 0, 0), 2)
            cv2.line(image, (box.pt4x, box.pt4y), (box.pt1x, box.pt1y), (255, 0, 0), 2)
            
            # 绘制类别和置信度
            title = f"{self.CLASSES[box.classId]} {box.score:.2f}"
            cv2.putText(image, title, (box.pt1x, box.pt1y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)
            
        return image
        
    def release(self):
        """
        释放RKNN资源
        在不再使用检测器时调用此方法
        """
        if hasattr(self, 'rknn'):
            self.rknn.release()
            self.rknn = None

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.release()

# 辅助类定义
class CSXYWHR:
    def __init__(self, classId, score, x, y, w, h, angle):
        self.classId = classId
        self.score = score
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle

class DetectBox:
    def __init__(self, classId, score, pt1x, pt1y, pt2x, pt2y, pt3x, pt3y, pt4x, pt4y, angle):
        self.classId = classId
        self.score = score
        self.pt1x = pt1x
        self.pt1y = pt1y
        self.pt2x = pt2x
        self.pt2y = pt2y
        self.pt3x = pt3x
        self.pt3y = pt3y
        self.pt4x = pt4x
        self.pt4y = pt4y
        self.angle = angle

if __name__ == '__main__':
    # 示例用法
    parser = argparse.ArgumentParser(description='RKNN YOLO Demo')
    parser.add_argument('--model_path', type=str, required=True, help='model path')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    args = parser.parse_args()
    
    # 创建检测器实例
    detector = RKNN_YOLO(args.model_path, args.target, args.device_id)
    
    # 读取测试图像
    image = cv2.imread('test.jpg')
    if image is None:
        print('Failed to load test image')
        sys.exit(-1)
        
    # 执行检测
    boxes = detector.detect(image)
    
    # 绘制结果
    result_image = detector.draw_result(image.copy(), boxes)
    
    # 保存结果
    cv2.imwrite('result.jpg', result_image)
    print('Detection result saved to result.jpg')