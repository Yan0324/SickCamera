"""
@Description :   this moudle is used to control the sick vision device and get the data.
                 The common module in the folder is required
@Author      :   Cao Yingjie
@Time        :   2025/04/23 08:47:44
"""

from common.Control import Control
from common.Streaming import Data
from common.Stream import Streaming
from common.Streaming.BlobServerConfiguration import BlobClientConfig
import cv2
import numpy as np
import time
import logging
import socket

class QtVisionSick:
    """
    西克相机控制类
    用于获取相机的强度图数据
    """
    
    def __init__(self, ipAddr="192.168.10.5", port=2122, protocol="Cola2"):
        """
        初始化西克相机
        
        Args:
            ipAddr (str): 相机IP地址
            port (int): 相机控制端口
            protocol (str): 通信协议
        """
        self.ipAddr = ipAddr
        self.control_port = port  # 控制端口
        self.streaming_port = 2114  # 数据流端口
        self.protocol = protocol
        self.deviceControl = None
        self.streaming_device = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        self.camera_params = None  # 存储相机参数
        self.use_single_step = True  # 默认使用单步模式
        
    def _check_camera_available(self):
        """
        检查相机是否可访问
        
        Returns:
            bool: 相机是否可访问
        """
        try:
            # 创建socket连接测试
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 设置超时时间为2秒
            result = sock.connect_ex((self.ipAddr, self.control_port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.error(f"Error checking camera availability: {str(e)}")
            return False
        
    def connect(self, max_retries=3, use_single_step=True):
        """
        连接相机并初始化流
        
        Args:
            max_retries (int): 最大重试次数
            use_single_step (bool): 是否使用单步模式
            
        Returns:
            bool: 连接是否成功
        """
        if not self._check_camera_available():
            self.logger.error(f"Camera at {self.ipAddr}:{self.control_port} is not accessible")
            return False
            
        self.use_single_step = use_single_step
            
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting to connect to camera (attempt {attempt + 1}/{max_retries})")
                
                # 创建设备控制实例
                self.deviceControl = Control(self.ipAddr, self.protocol, self.control_port)
                
                # 打开连接
                self.deviceControl.open()
                
                # 尝试登录 - 在连接时登录，保持登录状态
                try:
                    self.deviceControl.login(Control.USERLEVEL_SERVICE, 'CUST_SERV')
                    self.logger.info("以服务级别登录成功")
                except Exception as e:
                    self.logger.warning(f"Service level login failed, trying client level: {str(e)}")
                    self.deviceControl.login(Control.USERLEVEL_AUTH_CLIENT, 'CLIENT')
                    self.logger.info("以客户端级别登录成功")
                
                # 获取设备信息
                name, version = self.deviceControl.getIdent()
                self.logger.info(f"Connected to device: {name.decode('utf-8')}, version: {version.decode('utf-8')}")
                
                # 尝试设置较低的帧速率以减少延迟
                try:
                    # 获取当前帧周期 (微秒)
                    current_frame_period = self.deviceControl.getFramePeriodUs()
                    self.logger.info(f"当前帧周期: {current_frame_period} 微秒")
                    
                    # 设置较低的帧率 (例如 30 fps = 33333 微秒)
                    self.deviceControl.setFramePeriodUs(33333)
                    new_frame_period = self.deviceControl.getFramePeriodUs()
                    self.logger.info(f"设置新帧周期: {new_frame_period} 微秒")
                except Exception as e:
                    self.logger.warning(f"设置帧率失败: {str(e)}")
                
                # 配置流设置
                streamingSettings = BlobClientConfig()
                streamingSettings.setTransportProtocol(self.deviceControl, streamingSettings.PROTOCOL_TCP)
                streamingSettings.setBlobTcpPort(self.deviceControl, self.streaming_port)
                
                # 初始化流
                self.streaming_device = Streaming(self.ipAddr, self.streaming_port)
                self.streaming_device.openStream()
                
                # 根据模式决定流的处理方式
                if self.use_single_step:
                    self.logger.info("使用单步模式，先停止流并设置为单步模式")
                    # 确保流已停止
                    self.deviceControl.stopStream()
                    time.sleep(0.5)  # 等待相机完全停止流
                else:
                    self.logger.info("使用连续流模式，启动流")
                    self.deviceControl.startStream()
                    time.sleep(0.5)  # 等待流启动
                
                self.is_connected = True
                self.logger.info("Successfully connected to camera")
                return True
                
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                self.disconnect()
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待1秒后重试
                continue
                
        return False
            
    def disconnect(self):
        """断开相机连接并释放资源"""
        try:
            if self.is_connected:
                if self.deviceControl:
                    # 先停止流
                    try:
                        # 确保在停止流前先登录
                        try:
                            self.deviceControl.login(Control.USERLEVEL_AUTH_CLIENT, 'CLIENT')
                        except Exception as e:
                            self.logger.warning(f"登录设备时出错: {str(e)}")
                            
                        # 如果处于单步模式，先确保停止单步获取
                        if self.use_single_step:
                            try:
                                # 停止所有正在进行的单步操作
                                self.deviceControl.stopStream()
                                time.sleep(0.2)  # 等待相机处理命令
                                self.logger.info("单步模式已停止")
                            except Exception as e:
                                self.logger.warning(f"停止单步模式时出错: {str(e)}")
                        
                        # 停止数据流
                        self.deviceControl.stopStream()
                        time.sleep(0.2)  # 等待相机处理命令
                        self.logger.info("数据流已停止")
                    except Exception as e:
                        self.logger.warning(f"停止流时出错: {str(e)}")
                        
                    # 关闭流设备
                    if self.streaming_device:
                        try:
                            self.streaming_device.closeStream()
                            self.logger.info("流连接已关闭")
                        except Exception as e:
                            self.logger.warning(f"关闭流连接时出错: {str(e)}")
                    
                    # 登出设备
                    try:
                        self.deviceControl.logout()
                        self.logger.info("设备已登出")
                    except Exception as e:
                        self.logger.warning(f"登出设备时出错: {str(e)}")
                        
                    # 关闭控制连接
                    try:
                        self.deviceControl.close()
                        self.logger.info("控制连接已关闭")
                    except Exception as e:
                        self.logger.warning(f"关闭控制连接时出错: {str(e)}")
                        
                self.is_connected = False
                self.logger.info("相机连接已完全断开")
        except Exception as e:
            self.logger.error(f"断开连接时出错: {str(e)}")
        finally:
            # 确保所有引用都被清除
            self.deviceControl = None
            self.streaming_device = None
            
    def get_frame_no_step(self):
        """
        获取当前帧数据，不发送单步命令
        适用于单步模式下，单步命令已经在外部发送过的情况
        
        Returns:
            tuple: (success, depth_data, intensity_image)
                success (bool): 是否成功获取数据
                depth_data (list): 深度图数据
                intensity_image (numpy.ndarray): 强度图
        """
        if not self.is_connected:
            self.logger.error("相机未连接")
            return False, None, None
            
        try:
            # 如果是单步模式，先发送单步命令
            if self.use_single_step:
                try:
                    self.deviceControl.login(Control.USERLEVEL_AUTH_CLIENT, 'CLIENT')
                    self.deviceControl.trigger()
                    time.sleep(0.1)  # 等待相机处理触发命令
                except Exception as e:
                    self.logger.error(f"发送单步命令失败: {str(e)}")
                    return False, None, None
            
            # 设置更长的超时时间
            self.streaming_device.socket.settimeout(5.0)  # 设置5秒超时
            
            # 获取帧数据
            self.streaming_device.getFrame()
            wholeFrame = self.streaming_device.frame
            
            # 解析数据
            myData = Data.Data()
            myData.read(wholeFrame)
            
            if not myData.hasDepthMap:
                self.logger.warning("No depth map data available")
                return False, None, None
                
            # 获取深度数据
            distance_data = list(myData.depthmap.distance)
            
            # 获取强度数据
            intensityData = list(myData.depthmap.intensity)
            numCols = myData.cameraParams.width
            numRows = myData.cameraParams.height
            
            # 重塑数据为图像
            image = np.array(intensityData).reshape((numRows, numCols))
            
            # 直接调整对比度，不进行归一化
            adjusted_image = cv2.convertScaleAbs(image, alpha=0.05, beta=1)
            
            # 保存相机参数
            self.camera_params = myData.cameraParams
            
            return True, distance_data, adjusted_image
            
        except Exception as e:
            self.logger.error(f"Error getting frame without step: {str(e)}")
            return False, None, None
            
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()
        
    def __del__(self):
        """确保在销毁时断开连接"""
        self.disconnect()

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 使用示例
    with QtVisionSick() as camera:
        if camera.is_connected:
            success, depth_data, intensity_image = camera.get_frame_no_step()
            if success:
                # 保存图像
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"intensity_{timestamp}.jpg", intensity_image)
                print(f"Image saved as intensity_{timestamp}.jpg")
            else:
                print("Failed to get depth and intensity data")
        else:
            print("Failed to connect to camera")

  
  

