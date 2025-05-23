import json
import logging
from typing import Any, Callable, Optional, Dict, List
import paho.mqtt.client as mqtt
from paho.mqtt.properties import Properties
from paho.mqtt.packettypes import PacketTypes
import time

# MQTT 服务类
class MQTTService:
    # 连接状态枚举类
    class ConnectionState:
        DISCONNECTED = 0  # 断开连接
        CONNECTING = 1     # 正在连接
        CONNECTED = 2      # 已连接
        RECONNECTING = 3   # 正在重新连接

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):

        # 验证配置
        self._validate_config(config)
        
        # 初始化成员变量
        self.config = config
        self.logger = logger or logging.getLogger(__name__)  # 日志记录器
        self._client = None  # MQTT 客户端
        self._connection_state = self.ConnectionState.DISCONNECTED  # 连接状态
        self._reconnect_attempts = 0  # 重新连接尝试次数
        self._max_reconnect_attempts = 5  # 最大重新连接尝试次数
        self._on_message_received: Optional[Callable[[str, str], None]] = None  # 消息接收回调函数
        
        # 初始化客户端
        self._init_client()

    def _validate_config(self, config: Dict[str, Any]):
        required_keys = ['broker', 'port', 'client_id']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

    # 初始化 MQTT 客户端
    def _init_client(self):
        self._client = mqtt.Client(
            client_id=self.config['client_id'],  # 客户端 ID
            protocol=mqtt.MQTTv5  # 使用 MQTT v5 协议
        )
        self._client.on_connect = self._on_connect  # 设置连接回调函数
        self._client.on_disconnect = self._on_disconnect  # 设置断开连接回调函数
        self._client.on_message = self._on_message  # 设置消息接收回调函数

    def connect(self):
        """同步连接方法"""
        if self._connection_state == self.ConnectionState.CONNECTED:
            self.logger.warning("Already connected to MQTT broker")
            return

        self._connection_state = self.ConnectionState.CONNECTING
        try:
            # 同步连接
            self._client.connect(
                host=self.config['broker'],
                port=self.config['port'],
                keepalive=self.config.get('keepalive', 60)
            )
            
            # 启动网络循环
            self._client.loop_start()
            
            # 等待连接建立
            retry = 0
            while not self._client.is_connected() and retry < 5:
                time.sleep(0.5)
                retry += 1

            if not self._client.is_connected():
                raise ConnectionError("Failed to establish MQTT connection")

        except Exception as e:
            self._client.loop_stop()
            self.logger.error(f"Connection failed: {str(e)}")
            raise


    def subscribe(self, topic: str, qos: int = 1) -> None:
        """动态订阅单个主题"""
        try:
            result = self._client.subscribe(topic, qos)
            self.logger.info(f"订阅成功: {topic} (QoS:{qos}, mid:{result[1]})")
        except Exception as e:
            self.logger.error(f"订阅失败: {topic} - {str(e)}")
            raise

    def unsubscribe(self, topic: str) -> None:
        """取消订阅"""
        try:
            result = self._client.unsubscribe(topic)
            self.logger.info(f"取消订阅: {topic} (mid:{result[1]})")
        except Exception as e:
            self.logger.error(f"取消订阅失败: {topic} - {str(e)}")
            raise

    # 发布消息
    def publish(self, topic: str, payload: Any, qos: int = 1) -> None:
        # 检查是否已连接
        if not self.is_connected:
            raise RuntimeError("MQTT 客户端未连接")
        
        try:
            # 将 payload 转换为字符串或字节类型
            if not isinstance(payload, (bytes, str)):
                payload = json.dumps(payload)
            
            # 设置发布属性
            pub_props = Properties(PacketTypes.PUBLISH)
            pub_props.MessageExpiryInterval = 3600  # 消息过期时间
            
            # 发布消息
            result = self._client.publish(
                topic=topic,  # 主题
                payload=payload,  # 消息内容
                qos=qos,  # QoS 等级
                properties=pub_props,  # 消息属性
                retain=False  # 是否保留消息
            )
            
            # 等待消息发布完成
            result.wait_for_publish()
            self.logger.debug("已发布消息到 %s [%d 字节]", topic, len(payload))
        
        except Exception as e:
            # 发布失败，记录错误日志
            self.logger.error("发布消息失败：%s", str(e))
            raise

    @property
    def is_connected(self) -> bool:

        return self._client.is_connected()

    #回调函数，在 MQTT 客户端与代理建立连接后被自动触发调用。
    def _on_connect(self, client, userdata, flags, rc, properties=None):

        if rc == 0:
            self._connection_state = self.ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self.logger.info("MQTT connection established")
        else:
            self.logger.error("Connection failed with code: %d", rc)

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self._connection_state = self.ConnectionState.DISCONNECTED
        self.logger.warning("Disconnected from MQTT broker (code: %d)", rc)
        
        if rc != 0 and self._reconnect_attempts < self._max_reconnect_attempts:
            self._handle_reconnection()

    def _handle_reconnection(self):
        self._connection_state = self.ConnectionState.RECONNECTING
        self._reconnect_attempts += 1
        
        delay = min(5 * self._reconnect_attempts, 30)
        self.logger.info("Attempting reconnect #%d (delay: %ds)", 
                       self._reconnect_attempts, delay)
        
        time.sleep(delay)
        self.connect()

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            if self._on_message_received:
                self._on_message_received(msg.topic, payload)
            self.logger.debug("Received message from %s [%d bytes]", 
                            msg.topic, len(payload))
        except Exception as e:
            self.logger.error("Error processing message: %s", str(e))

    def set_message_handler(self, callback: Callable[[str, str], None]):

        self._on_message_received = callback

    def disconnect(self):
        if self.is_connected:
            self._client.disconnect()
            self._client.loop_stop()
            self.logger.info("Gracefully disconnected from MQTT broker")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
