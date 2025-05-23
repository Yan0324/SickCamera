"""
@Description :   相机检测线程模块
@Author      :   Cao Yingjie
@Time        :   2025/04/23 08:47:37
"""

import cv2
import numpy as np
import time
import sys
import queue
import threading
from common.Streaming import Data
from utils.Coordinate_conversion import CoordinateTransformer

def camera_detection_thread(camera, detector, transformer, result_queue, exit_flag, processed_points=None, timing_stats=None):
    """
    相机检测线程函数，带有耗时统计功能
    Args:
        camera: 相机对象
        detector: 检测器对象
        transformer: 坐标转换器对象
        result_queue: 结果队列
        exit_flag: 退出标志
        processed_points: 已处理点列表，用于避免重复检测
        timing_stats: 时间统计字典
    """
    print("Camera detection thread started")
    
    # 连接断开计数和重连机制
    network_error_count = 0
    max_network_errors = 3
    reconnect_delay = 2  # 初始重连延迟（秒）
    max_reconnect_delay = 10  # 最大重连延迟（秒）
    
    # 初始化用于存储数据的对象
    myData = Data.Data()
    
    # 图像处理相关变量
    cycle_count = 0  # 帧计数器
    
    # 启动相机流模式
    try:
        # 确保流已启动
        camera.deviceControl.startStream()
        print("Camera stream started")
        
        # 等待流启动
        time.sleep(0.5)
        
        while not exit_flag.is_set():
            try:
                # 检查相机是否仍然连接
                if not camera.is_connected:
                    print("Camera connection lost, trying to reconnect...")
                    camera.disconnect()  # 确保完全断开
                    time.sleep(reconnect_delay)
                    if camera.connect():
                        print("Camera reconnected successfully")
                        # 重新启动流
                        camera.deviceControl.startStream()
                        print("Camera stream restarted")
                        # 重置网络错误计数
                        network_error_count = 0
                        reconnect_delay = 2  # 重置重连延迟
                    else:
                        print("Camera reconnection failed")
                        network_error_count += 1
                        # 增加重连延迟，但不超过最大值
                        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                        continue  # 重连失败，跳过本次循环
                
                # 获取图像开始时间
                image_acquisition_start = time.time()
                
                # 获取相机图像数据（流模式）
                try:
                    # 参考visionary_StreamingDemo.py中的流模式获取
                    camera.streaming_device.getFrame()
                    myData.read(camera.streaming_device.frame)
                except Exception as e:
                    print(f"Error getting frame data: {str(e)}")
                    network_error_count += 1
                    if network_error_count > max_network_errors:
                        print("Maximum network errors reached, trying to reconnect")
                        camera.disconnect()
                        time.sleep(reconnect_delay)
                        if camera.connect():
                            print("Camera reconnected successfully")
                            camera.deviceControl.startStream()
                            time.sleep(0.5)
                            network_error_count = 0
                            reconnect_delay = 2  # 重置重连延迟
                        else:
                            print("Camera reconnection failed")
                            # 增加重连延迟，但不超过最大值
                            reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)
                    time.sleep(0.5)
                    continue
                
                # 计算图像获取耗时
                image_acquisition_time = time.time() - image_acquisition_start
                if timing_stats:
                    timing_stats.setdefault('image_acquisition', []).append(image_acquisition_time)
                    
                print(f"Image acquisition time: {image_acquisition_time:.2f}s")
                
                if not myData.hasDepthMap:
                    print("No depth map data available")
                    time.sleep(0.1)
                    continue
                    
                # 获取深度数据
                depth_data = list(myData.depthmap.distance)
                
                # 获取强度数据
                intensityData = list(myData.depthmap.intensity)
                numCols = myData.cameraParams.width
                numRows = myData.cameraParams.height
                
                # 重塑数据为图像
                intensity_image = np.array(intensityData).reshape((numRows, numCols))
                
                # 直接调整对比度，不进行归一化
                intensity_image = cv2.convertScaleAbs(intensity_image, alpha=0.05, beta=1)
                
                # 保存相机参数
                camera.camera_params = myData.cameraParams
                
                # 开始目标检测计时
                detection_start = time.time()
                
                # 执行检测
                boxes = detector.detect(intensity_image)
                
                # 计算目标检测耗时
                detection_time = time.time() - detection_start
                if timing_stats:
                    timing_stats.setdefault('object_detection', []).append(detection_time)
                    
                print(f"Object detection time: {detection_time:.2f}s")
                
                # 绘制图像和检测框
                display_image = intensity_image.copy()
                # 转换为彩色图像以便绘制彩色框体
                if len(display_image.shape) == 2:
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
                
                # 获取原始图像尺寸
                img_height, img_width = display_image.shape[:2]
                
                # 增强图像对比度以便更好地显示
                # 先标准化到0-255
                if display_image.dtype != np.uint8:
                    normalized_image = cv2.normalize(display_image, None, 0, 255, cv2.NORM_MINMAX)
                    display_image = normalized_image.astype(np.uint8)
                
                # 应用直方图均衡化以增强对比度
                if not myData.hasDepthMap:  # 如果未成功获取图像，显示纯黑图像
                    display_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                else:
                    # 将强度图均衡化并转换为彩色
                    enhanced_intensity = cv2.equalizeHist(intensity_image)
                    display_image = cv2.cvtColor(enhanced_intensity, cv2.COLOR_GRAY2BGR)
                
                # 添加标题栏
                title_bar_height = 30
                title_bar = np.zeros((title_bar_height, display_image.shape[1], 3), dtype=np.uint8)
                title_bar[:] = (50, 50, 50)  # 深灰色背景
                
                # 在标题栏添加标题
                title_text = "SICK Visionary Detection System"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(title_text, font, 0.7, 2)[0]
                text_x = (display_image.shape[1] - text_size[0]) // 2
                cv2.putText(title_bar, title_text, (text_x, 20), font, 0.7, (255, 255, 255), 2)
                
                # 存储所有有效的检测点
                valid_points = []
                
                # 开始坐标转换计时
                coordinate_start = time.time()
                
                # 在原始图像上处理检测结果（在添加标题和信息栏之前）
                for box in boxes:
                    # 计算检测框中心点
                    center_x = int((box.pt1x + box.pt2x + box.pt3x + box.pt4x) / 4)
                    center_y = int((box.pt1y + box.pt2y + box.pt3y + box.pt4y) / 4)
                    
                    # 在图像上绘制检测框和类别信息
                    pt1 = (int(box.pt1x), int(box.pt1y))
                    pt2 = (int(box.pt2x), int(box.pt2y))
                    pt3 = (int(box.pt3x), int(box.pt3y))
                    pt4 = (int(box.pt4x), int(box.pt4y))
                    
                    # 绘制矩形框（使用不同颜色区分不同类别）
                    color = (0, 255, 0)  # 默认绿色
                    # 可选：根据类别ID设置不同颜色
                    if box.classId == 0:
                        color = (0, 255, 0)  # 绿色
                    elif box.classId == 1:
                        color = (255, 0, 0)  # 蓝色
                    elif box.classId == 2:
                        color = (0, 0, 255)  # 红色
                    
                    # 绘制四边形
                    cv2.line(display_image, pt1, pt2, color, 2)
                    cv2.line(display_image, pt2, pt3, color, 2)
                    cv2.line(display_image, pt3, pt4, color, 2)
                    cv2.line(display_image, pt4, pt1, color, 2)
                    
                    # 绘制中心点
                    cv2.circle(display_image, (center_x, center_y), 3, (0, 255, 255), -1)
                    
                    # 绘制类别和置信度
                    score_text = f"{detector.CLASSES[box.classId]} {box.score:.2f}"
                    cv2.putText(display_image, score_text, (pt1[0], pt1[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # 计算3D坐标
                    success_coord, coords = calculate_3d_coordinates_from_depth(
                        center_x, center_y, 
                        depth_data, 
                        camera.camera_params
                    )
                    
                    if success_coord:
                        x_cam, y_cam, z = coords
                        
                        # 转换到机器人坐标系
                        robot_coords = transformer.transform_point(np.array([x_cam, y_cam, z]))
                        
                        # 在图像上绘制坐标信息
                        coord_text = f"X:{robot_coords[0]:.1f} Y:{robot_coords[1]:.1f} Z:{robot_coords[2]:.1f}"
                        cv2.putText(display_image, coord_text, (pt1[0], pt1[1] + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # 存储有效的检测点信息
                        valid_points.append({
                            'coords': robot_coords,
                            'camera_coords': (x_cam, y_cam, z),
                            'depth': z,
                            'box': box,
                            'center': (center_x, center_y)
                        })
                
                # 计算坐标转换耗时
                coordinate_time = time.time() - coordinate_start
                if timing_stats:
                    timing_stats.setdefault('coordinate_transform', []).append(coordinate_time)
                    
                print(f"Coordinate transformation time: {coordinate_time:.2f}s")
                
                # 为有效点添加索引编号
                for i, point in enumerate(valid_points):
                    center = point['center']
                    cv2.putText(display_image, f"#{i+1}", 
                                (center[0] + 5, center[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # 在图像上方添加标题
                title_height = 30
                title_bar = np.ones((title_height, img_width, 3), dtype=np.uint8) * 30  # 暗灰色背景
                title_text = "SICK Visionary Detection System"
                cv2.putText(title_bar, title_text, (img_width//2 - 150, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # 将标题栏添加到图像顶部
                display_image_with_title = np.vstack((title_bar, display_image))
                
                # 创建信息栏，放在图像下方
                info_bar_height = 60
                info_bar = np.ones((info_bar_height, img_width, 3), dtype=np.uint8) * 50  # 深灰色背景
                
                # 添加时间和帧率信息
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
                
                cv2.putText(info_bar, current_time_str, (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(info_bar, f"Frame: {cycle_count}", (10, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 添加队列状态信息
                queue_info = f"Queue Size: {result_queue.qsize()}/{result_queue.maxsize}"
                cv2.putText(info_bar, queue_info, (img_width - 150, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 添加本帧处理时间信息
                frame_time_info = f"Detection: {detection_time*1000:.1f}ms | Transform: {coordinate_time*1000:.1f}ms"
                cv2.putText(info_bar, frame_time_info, (img_width - 400, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 添加操作提示
                controls_info = "Q: Exit | S: Save"
                cv2.putText(info_bar, controls_info, (img_width - 220, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 合并图像和信息栏
                combined_image = np.vstack((display_image_with_title, info_bar))
                
                # 如果需要，缩放图像以适应屏幕
                scale_percent = 100  # 默认不缩放
                if img_width > 1200:  # 如果图像太宽，进行缩放
                    scale_percent = 75
                
                if scale_percent != 100:
                    width = int(combined_image.shape[1] * scale_percent / 100)
                    height = int(combined_image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    combined_image = cv2.resize(combined_image, dim, interpolation=cv2.INTER_AREA)
                
                # 确保图像是有效的8位格式
                if combined_image.dtype != np.uint8:
                    combined_image = combined_image.astype(np.uint8)
                
                # 确保窗口已创建
                try:
                    # 显示图像
                    cv2.imshow("SICK Camera Detection", combined_image)
                    
                    # 更新窗口，防止窗口无响应
                    cv2.setWindowProperty("SICK Camera Detection", cv2.WND_PROP_VISIBLE, 1)
                    
                    # 每100帧保存一次图像，用于调试
                    if cycle_count % 100 == 0:
                        debug_filename = f"debug_frame_{cycle_count}.jpg"
                        cv2.imwrite(debug_filename, combined_image)
                        print(f"Debug image saved: {debug_filename}")
                    
                    key = cv2.waitKey(1)  # 1毫秒刷新，保持窗口更新
                except Exception as e:
                    print(f"Window display error: {str(e)}")
                    key = -1  # 设置一个无效键值
                
                # 检查是否按下q键退出
                if key == ord('q'):
                    print("Key 'q' detected, preparing to exit...")
                    exit_flag.set()
                    break
                
                # 检查是否按下s键保存图像
                if key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, combined_image)
                    print(f"Image saved as {filename}")
                
                # 添加帧计数
                cycle_count += 1
                
                # 开始结果处理计时
                results_process_start = time.time()
                
                # 检查是否有有效点
                if valid_points:
                    # 找出最近的点
                    nearest_point = min(valid_points, key=lambda p: p['depth'])
                    
                    # 将最佳点放入队列
                    detection_info = {
                        'coords': nearest_point['coords'],
                        'class_id': nearest_point['box'].classId,
                        'score': nearest_point['box'].score,
                        'timestamp': time.time()
                    }
                    
                    # 检查队列中是否已经有相似的点
                    duplicate_found = False
                    same_point_threshold = 10.0  # 相同点判定阈值（mm）
                    
                    # 将当前队列中的所有点取出来检查
                    temp_items = []
                    print("\nCurrent queue status (detection thread):")
                    queue_empty = result_queue.empty()
                    print(f"Queue empty: {queue_empty}")
                    
                    item_count = 0
                    while not result_queue.empty():
                        try:
                            item = result_queue.get_nowait()
                            item_count += 1
                            temp_items.append(item)
                            
                            # 打印队列中的点
                            print(f"Point {item_count} in queue: X={item['coords'][0]:.2f}, Y={item['coords'][1]:.2f}, Z={item['coords'][2]:.2f}, Class={detector.CLASSES[item['class_id']]}, Timestamp={item['timestamp']:.2f}")
                            
                            # 计算与新点的距离
                            existing_point = item['coords']
                            new_point = detection_info['coords']
                            point_distance = np.sqrt(
                                (existing_point[0] - new_point[0])**2 +
                                (existing_point[1] - new_point[1])**2 +
                                (existing_point[2] - new_point[2])**2
                            )
                            
                            # 如果距离小于阈值，判定为重复点
                            if point_distance < same_point_threshold:
                                duplicate_found = True
                                print(f"Duplicate point detected, distance to existing point: {point_distance:.2f}mm, skipping")
                                print(f"  Existing point: X={existing_point[0]:.2f}, Y={existing_point[1]:.2f}, Z={existing_point[2]:.2f}")
                                print(f"  New point: X={new_point[0]:.2f}, Y={new_point[1]:.2f}, Z={new_point[2]:.2f}")
                        except queue.Empty:
                            break
                    
                    # 将原来的点放回队列
                    for item in temp_items:
                        try:
                            result_queue.put_nowait(item)
                        except queue.Full:
                            # 如果队列已满，丢弃最早的点
                            print("Queue is full, dropping oldest point")
                            break
                    
                    print(f"Check complete: {item_count} points in queue, current point is{' ' if duplicate_found else ' not '}a duplicate")
                    
                    # 只有非重复点才添加到队列
                    if not duplicate_found:
                        # 检查是否与已处理点太接近
                        is_processed = False
                        if processed_points is not None:
                            new_point = detection_info['coords']
                            for proc_point in processed_points:
                                point_distance = np.sqrt(
                                    (proc_point[0] - new_point[0])**2 +
                                    (proc_point[1] - new_point[1])**2 +
                                    (proc_point[2] - new_point[2])**2
                                )
                                if point_distance < same_point_threshold:
                                    is_processed = True
                                    print(f"Point already processed, distance to processed point: {point_distance:.2f}mm, skipping")
                                    print(f"  New point: X={new_point[0]:.2f}, Y={new_point[1]:.2f}, Z={new_point[2]:.2f}")
                                    print(f"  Processed point: X={proc_point[0]:.2f}, Y={proc_point[1]:.2f}, Z={proc_point[2]:.2f}")
                                    break
                        
                        # 尝试放入队列，如果队列满则移除最早的元素
                        if not is_processed:
                            try:
                                if result_queue.full():
                                    # 移除最早的元素
                                    try:
                                        result_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                result_queue.put_nowait(detection_info)
                                print(f"New target detected, coordinates: X={detection_info['coords'][0]:.2f}, Y={detection_info['coords'][1]:.2f}, Z={detection_info['coords'][2]:.2f}")
                            except queue.Full:
                                print("Result queue is full, cannot add new detection")
                else:
                    # 没有检测到有效点时，清空队列
                    print("No valid points detected, clearing queue")
                    
                    # 清空队列
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    print(f"Queue cleared, current size: {result_queue.qsize()}")
                
                # 计算结果处理耗时
                results_process_time = time.time() - results_process_start
                if timing_stats:
                    timing_stats.setdefault('results_processing', []).append(results_process_time)
                    
                print(f"Results processing time: {results_process_time:.2f}s")
                
                # 计算总帧处理时间
                total_frame_time = time.time() - image_acquisition_start
                if timing_stats:
                    timing_stats.setdefault('total_frame_processing', []).append(total_frame_time)
                    
                print(f"Total frame processing time: {total_frame_time:.2f}s (FPS: {1/total_frame_time:.2f})")
                
                # 适当休眠，避免CPU占用过高
                time.sleep(0.01)
                
                # 添加帧计数
                cycle_count += 1
                
            except Exception as e:
                print(f"Camera detection thread error: {str(e)}")
                time.sleep(1)  # 出错后短暂暂停
    
    except Exception as e:
        print(f"启动相机流失败: {str(e)}")
    
    finally:
        # 停止相机流
        try:
            camera.deviceControl.stopStream()
            print("Camera stream stopped")
            # 关闭流连接
            camera.streaming_device.closeStream()
            print("Stream connection closed")
        except Exception as e:
            print(f"Failed to stop camera stream: {str(e)}")
            
        print("Camera detection thread exited")

def calculate_3d_coordinates_from_depth(x, y, depth_data, camera_params):
    """
    从深度数据计算3D坐标
    Args:
        x: 图像x坐标
        y: 图像y坐标
        depth_data: 深度数据
        camera_params: 相机参数
    Returns:
        tuple: (success, (x_cam, y_cam, z))
            success (bool): 是否成功计算坐标
            x_cam, y_cam, z: 相机坐标系下的3D坐标
    """
    try:
        # 检查输入参数
        if depth_data is None or camera_params is None:
            return False, (0, 0, 0)
            
        # 检查坐标是否在有效范围内
        if x < 0 or x >= camera_params.width or y < 0 or y >= camera_params.height:
            return False, (0, 0, 0)
            
        # 计算索引
        index = y * camera_params.width + x
        if index >= len(depth_data):
            return False, (0, 0, 0)
            
        # 获取深度值
        z = depth_data[index]
        
        # 检查深度值是否有效
        if z <= 0:
            return False, (0, 0, 0)
            
        # 计算相机坐标系下的x和y坐标
        xp = (camera_params.cx - x) / camera_params.fx
        yp = (camera_params.cy - y) / camera_params.fy
        
        # 计算径向畸变
        r2 = (xp * xp + yp * yp)
        r4 = r2 * r2
        k = 1 + camera_params.k1 * r2 + camera_params.k2 * r4
        
        xd = xp * k
        yd = yp * k
        
        # 计算相机坐标系下的坐标
        s0 = np.sqrt(xd*xd + yd*yd + 1)
        x_cam = xd * z / s0
        y_cam = yd * z / s0
        z_cam = z / s0 - camera_params.f2rc
        
        # 转换到世界坐标系
        m_c2w = np.array(camera_params.cam2worldMatrix).reshape(4, 4)
        x_world = (m_c2w[0, 3] + z_cam * m_c2w[0, 2] + y_cam * m_c2w[0, 1] + x_cam * m_c2w[0, 0])
        y_world = (m_c2w[1, 3] + z_cam * m_c2w[1, 2] + y_cam * m_c2w[1, 1] + x_cam * m_c2w[1, 0])
        z_world = (m_c2w[2, 3] + z_cam * m_c2w[2, 2] + y_cam * m_c2w[2, 1] + x_cam * m_c2w[2, 0])
        
        return True, (x_world, y_world, z_world)
    except Exception as e:
        return False, (0, 0, 0) 