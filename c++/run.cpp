// 标准库头文件
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>
#include <iterator>
#include <chrono>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>

// OpenCV库头文件
#include <opencv2/opencv.hpp>

// Visionary相机相关头文件
#include "VisionaryControl.h"
#include "CoLaParameterReader.h"
#include "CoLaParameterWriter.h"
#include "VisionaryTMiniData.h" // 用于Time of Flight数据的特定头文件
#include "VisionaryDataStream.h"
#include "PointCloudPlyWriter.h"

// 自定义头文件
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "PointXYZ.h"

// MQTT相关头文件
#include <cstdlib>
#include <string>
#include <cstring>
#include <cctype>
#include "mqtt/async_client.h"
#include <random>
#include <ctime>
#include <cstdlib>
#include <cjson/cJSON.h>

// MQTT配置参数
const std::string SERVER_ADDRESS("192.168.0.164:1883");  // MQTT服务器地址
const std::string CLIENT_ID("rk3588");                   // 客户端ID
const std::string DETECTOR_TOPIC("camera/start_detector"); // 检测器启动主题
const std::string HEIGHT_TOPIC("camera/lifting_height");   // 高度测量主题
const std::string GRAB_TOPIC("warehouse/start_grab");      // 抓取启动主题

const int QOS = 2;                    // MQTT服务质量级别
const int N_RETRY_ATTEMPTS = 5;       // MQTT重连尝试次数
bool CAMERA_RUN = false;              // 相机运行状态标志
bool LEFTING_HEIGHT = false;          // 高度测量状态标志
bool GRAB = false;                    // 抓取状态标志

using namespace visionary;

#pragma region own Methods
// 箱子信息结构体，用于存储检测到的箱子相关信息
struct BoxInfo
{
  bool isTrapezoid;   // 是否为梯形
  int detectionCount; // 检测数量
  int topLayerCount;  // 最高层个数
  int layerHeight;    // 层数
  // 构造函数，方便初始化
  BoxInfo(bool isTrap, int count, int topCount, float height)
      : isTrapezoid(isTrap), detectionCount(count), topLayerCount(topCount), layerHeight(height) {}
};

// 检测框信息结构体，用于存储检测结果
struct InfoBox
{
  int detectNum;                           // 检测到的目标数量
  std::vector<float16_t> detectBoxAverage; // 检测框的平均值数组
  InfoBox(int detectNum, std::vector<float16_t> detectBoxAverage)
      : detectNum(detectNum), detectBoxAverage(detectBoxAverage) {}
};

// 生成指定长度的随机字符串
std::string generateRandomString(size_t length)
{
  const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::string result;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, characters.size() - 1);

  for (size_t i = 0; i < length; ++i)
  {
    result += characters[dis(gen)];
  }

  return result;
}

// 将BoxInfo结构体转换为JSON字符串
std::string toJsonString(const struct BoxInfo *box)
{
  // 创建 cJSON 对象
  cJSON *root = cJSON_CreateObject();

  // 添加结构体属性到 JSON
  cJSON_AddBoolToObject(root, "isTrapezoid", box->isTrapezoid);
  cJSON_AddNumberToObject(root, "detectionCount", box->detectionCount);
  cJSON_AddNumberToObject(root, "topLayerCount", box->topLayerCount);
  cJSON_AddNumberToObject(root, "layerHeight", box->layerHeight);

  // 将 cJSON 对象转换为字符串
  char *jsonStr = cJSON_Print(root);
  std::string result(jsonStr);

  // 释放内存
  cJSON_Delete(root);
  free(jsonStr);

  return result;
}

// 将InfoBox结构体转换为cJSON对象
cJSON *InfoBoxToJson(const InfoBox &infoBox)
{
  cJSON *root = cJSON_CreateObject();

  // 添加 detectNum 字段
  cJSON_AddNumberToObject(root, "detectNum", infoBox.detectNum);

  // 创建 detectBoxAverage 数组
  cJSON *averageArray = cJSON_CreateArray();
  for (const auto &value : infoBox.detectBoxAverage)
  {
    cJSON_AddItemToArray(averageArray, cJSON_CreateNumber(value));
  }
  cJSON_AddItemToObject(root, "detectBoxAverage", averageArray);

  return root;
}

// 将 InfoBox 转换为 JSON 字符串
std::string InfoBoxToJsonString(const InfoBox &infoBox)
{
  cJSON *json = InfoBoxToJson(infoBox);
  char *jsonStr = cJSON_Print(json);
  std::string result(jsonStr);

  // 释放内存
  cJSON_free(jsonStr);
  cJSON_Delete(json);

  return result;
}

// 生成随机图片文件名
std::string generateRandomImageName(const std::string &extension, size_t nameLength = 10)
{
  std::string name = generateRandomString(nameLength);
  return name + extension;
}

// 计算检测框中心区域的均值
void calculateCenterMean(const cv::Mat &image, const std::vector<Detection> &objects)
{
  std::vector<double> meanValues;
  for (const auto &obj : objects)
  {
    // 定义源点集（检测框的四个角点）
    std::vector<cv::Point2f> srcPoints = {
        {obj.point1.x, obj.point1.y}, // 左上
        {obj.point2.x, obj.point2.y}, // 右上
        {obj.point3.x, obj.point3.y}, // 右下
        {obj.point4.x, obj.point4.y}  // 左下
    };

    // 定义目标点集（用于透视变换）
    int cropWidth = obj.w;
    int cropHeight = obj.h;
    std::vector<cv::Point2f> dstPoints = {
        {0, 0},                  // 左上
        {cropWidth, 0},          // 右上
        {cropWidth, cropHeight}, // 右下
        {0, cropHeight}          // 左下
    };

    // 计算透视变换矩阵
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    // 应用透视变换
    cv::Mat croppedImage;
    cv::warpPerspective(image, croppedImage, perspectiveMatrix, cv::Size(cropWidth, cropHeight));
    
    // 计算中心区域的均值
    int centerX = cropWidth / 2;
    int centerY = cropHeight / 2;
    int halfSize = 5;
    // 定义中心矩形的左上角和右下角
    cv::Rect centerRect(centerX - halfSize, centerY - halfSize, 10, 10);
    // 提取中心矩形区域
    cv::Mat centerRegion = image(centerRect);
    // 计算均值
    cv::Scalar meanValue = cv::mean(centerRegion);
    std::cout << meanValue[0] << std::endl;
  }
}

// 计算点云中指定区域的平均值
float16_t calculateAverage(const std::vector<PointXYZ> &distanceMap, int width, int height, int centerX, int centerY, int radius)
{
  int count = 0;
  double sum = 0;
  int startX = std::max(0, centerX - radius);
  int startY = std::max(0, centerY - radius);
  int endX = std::min(width, centerX + radius);
  int endY = std::min(height, centerY + radius);

  for (int i = startY; i < endY; ++i)
  {
    for (int j = startX; j < endX; ++j)
    {
      float distance = std::sqrt((j - centerX) * (j - centerX) + (i - centerY) * (i - centerY));
      if (distance <= radius)
      {
        int index = i * width + j;
        if (!isnan(distanceMap[index].z))
        {
          sum += distanceMap[index].z;
        }
        else
        {
          continue;
        }
        count++;
      }
    }
  }

  return count > 0 ? static_cast<float16_t>(sum / count) : 0.0;
}

// 从点云数据中提取Z值矩阵
std::vector<std::vector<float>> extractZValues(const std::vector<PointXYZ> &pointCloud, int width, int height)
{
  std::vector<std::vector<float>> zValues(height, std::vector<float>(width, 0.0f));

  for (int i = 0; i < height; ++i)
  {
    for (int j = 0; j < width; ++j)
    {
      int index = i * width + j;
      if (index < pointCloud.size())
      {
        zValues[i][j] = pointCloud[index].z;
      }
    }
  }

  return zValues;
}

// 计算正方形区域内的平均值
float calculateSquareAverage(const std::vector<std::vector<float>> &zValues, int width, int height, float centerX, float centerY, int radius)
{
  int count = 0;
  float sum = 0.0f;

  // 计算正方形区域的边界
  int startX = static_cast<int>(std::max(0.0f, centerX - radius));
  int startY = static_cast<int>(std::max(0.0f, centerY - radius));
  int endX = static_cast<int>(std::min(static_cast<float>(width), centerX + radius));
  int endY = static_cast<int>(std::min(static_cast<float>(height), centerY + radius));

  // 遍历正方形区域
  for (int i = startY; i < endY; ++i)
  {
    for (int j = startX; j < endX; ++j)
    {
      if (!isnan(zValues[i][j]))
      {
        sum += zValues[i][j];
      }
      else
      {
        continue;
      }
      count++;
    }
  }

  // 返回平均值
  return count > 0 ? sum / count : 0.0f;
}

// 将中心坐标和宽高转换为矩形的边界坐标
cv::Rect convertToRect(float x, float y, float w, float h)
{
  float halfW = w / 2.0f;
  float halfH = h / 2.0f;
  int left = static_cast<int>(x - halfW);
  int top = static_cast<int>(y - halfH);
  int right = static_cast<int>(x + halfW);
  int bottom = static_cast<int>(y + halfH);

  return cv::Rect(left, top, right - left, bottom - top);
}

// 计算两个矩形框的IoU（交并比）
float calculateIoU(const Detection &det1, const Detection &det2)
{
  cv::Rect rect1 = convertToRect(det1.x, det1.y, det1.w, det1.h);
  cv::Rect rect2 = convertToRect(det2.x, det2.y, det2.w, det2.h);

  // 计算交集区域
  cv::Rect inter = rect1 & rect2;
  float interArea = inter.area();

  // 计算并集区域
  float unionArea = rect1.area() + rect2.area() - interArea;

  // 计算IoU
  return interArea / unionArea;
}

// 处理重叠的检测框，移除重叠度高的低置信度框
void processDetections(std::vector<Detection> &detections, float iouThreshold)
{
  for (size_t i = 0; i < detections.size(); ++i)
  {
    for (size_t j = i + 1; j < detections.size(); ++j)
    {
      float iou = calculateIoU(detections[i], detections[j]);
      if (iou > iouThreshold)
      {
        // 如果IoU大于阈值，移除置信度较低的检测框
        if (detections[i].confidence < detections[j].confidence)
        {
          detections.erase(detections.begin() + i);
          --i; // 调整索引
          break;
        }
        else
        {
          detections.erase(detections.begin() + j);
          --j; // 调整索引
        }
      }
    }
  }
}
#pragma endregion

#pragma region mqtt structer
// MQTT动作监听器类，用于处理MQTT操作的成功和失败回调
class action_listener : public virtual mqtt::iaction_listener
{
  std::string name_;

  // 处理操作失败的回调函数
  void on_failure(const mqtt::token &tok) override
  {
    std::cout << name_ << " failure";
    if (tok.get_message_id() != 0)
      std::cout << " for token: [" << tok.get_message_id() << "]" << std::endl;
    std::cout << std::endl;
  }

  // 处理操作成功的回调函数
  void on_success(const mqtt::token &tok) override
  {
    std::cout << name_ << " success";
    if (tok.get_message_id() != 0)
      std::cout << " for token: [" << tok.get_message_id() << "]" << std::endl;
    auto top = tok.get_topics();
    if (top && !top->empty())
      std::cout << "\ttoken topic: '" << (*top)[0] << "', ..." << std::endl;
    std::cout << std::endl;
  }

public:
  action_listener(const std::string &name) : name_(name) {}
};

// MQTT回调处理类，用于处理MQTT连接、消息等事件
class callback : public virtual mqtt::callback,
                 public virtual mqtt::iaction_listener
{
  int nretry_;                    // 重连尝试次数计数器
  mqtt::async_client &cli_;       // MQTT客户端引用
  mqtt::connect_options &connOpts_; // MQTT连接选项
  action_listener subListener_;   // 订阅监听器

  // 重新连接到MQTT服务器
  void reconnect()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    try
    {
      cli_.connect(connOpts_, nullptr, *this);
    }
    catch (const mqtt::exception &exc)
    {
      std::cerr << "Error: " << exc.what() << std::endl;
      exit(1);
    }
  }

  // 处理连接失败的回调函数
  void on_failure(const mqtt::token &tok) override
  {
    std::cout << "Connection attempt failed" << std::endl;
    if (++nretry_ > N_RETRY_ATTEMPTS)
      exit(1);
    reconnect();
  }

  // 处理连接成功的回调函数
  void on_success(const mqtt::token &tok) override {}

  // 处理连接成功的回调函数
  void connected(const std::string &cause) override
  {
    std::cout << "\nConnection success" << std::endl;
    std::cout << "\nSubscribing to topic '" << DETECTOR_TOPIC << "'\n"
              << "\tfor client " << CLIENT_ID
              << " using QoS" << QOS << "\n"
              << "\nPress Q<Enter> to quit\n"
              << std::endl;

    // 订阅相关主题
    cli_.subscribe(DETECTOR_TOPIC, QOS, nullptr, subListener_);
    cli_.subscribe(HEIGHT_TOPIC, QOS, nullptr, subListener_);
    cli_.subscribe(GRAB_TOPIC, QOS, nullptr, subListener_);
  }

  // 处理连接丢失的回调函数
  void connection_lost(const std::string &cause) override
  {
    std::cout << "\nConnection lost" << std::endl;
    if (!cause.empty())
      std::cout << "\tcause: " << cause << std::endl;

    std::cout << "Reconnecting..." << std::endl;
    nretry_ = 0;
    reconnect();
  }

  // 处理消息到达的回调函数
  void message_arrived(mqtt::const_message_ptr msg) override
  {
    std::cout << "Message arrived" << std::endl;
    std::cout << "\ttopic: '" << msg->get_topic() << "'" << std::endl;
    std::cout << "\tpayload: '" << msg->to_string() << "'\n"
              << std::endl;

    // 根据不同的主题处理消息
    if (msg->get_topic() == "camera/start_detector")
    {
      CAMERA_RUN = true;
    }
    if (msg->get_topic() == "camera/lifting_height")
    {
      LEFTING_HEIGHT = true;
    }
    if (msg->get_topic() == "warehouse/start_grab")
    {
      GRAB = true;
    }
  }

  // 处理消息发送完成的回调函数
  void delivery_complete(mqtt::delivery_token_ptr token) override {}

public:
  callback(mqtt::async_client &cli, mqtt::connect_options &connOpts)
      : nretry_(0), cli_(cli), connOpts_(connOpts), subListener_("Subscription") {}
};
#pragma endregion

// 连接到相机并处理重连逻辑
bool connectToCamera(VisionaryDataStream &dataStream, const std::string &deviceIpAddr, unsigned short deviceBlobCtrlPort)
{
  const int MAX_RETRIES = 10;         // 最大重试次数
  const int RETRY_INTERVAL_MS = 5000; // 重试间隔时间（毫秒）

  int retries = 0;
  while (retries < MAX_RETRIES)
  {
    if (!dataStream.open(deviceIpAddr.c_str(), htons(deviceBlobCtrlPort)))
    {
      std::cerr << "Failed to open data stream connection to device. Retrying in "
                << RETRY_INTERVAL_MS / 1000 << " seconds..." << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
      retries++;
    }
    else
    {
      std::cout << "Successfully connected to the camera data stream." << std::endl;
      return true;
    }
  }
  std::cerr << "Failed to connect to the camera after " << MAX_RETRIES
            << " attempts. Exiting..." << std::endl;
  return false;
}

// YOLO检测函数（待实现）
std::vector<Detection> yoloDetection() {
  // TODO: 实现YOLO检测逻辑
};

int main(int argc, char *argv[])
{
  // 相机连接配置
  #pragma region camera_connect
  std::string deviceIpAddr("192.168.10.5");  // 相机IP地址
  unsigned short deviceBlobCtrlPort = 2114u; // 相机控制端口
  const char *model_file = argv[1];          // 模型文件路径
  std::cout << argv[1] << std::endl;
  std::cout << model_file << std::endl;

  // 创建Visionary实例
  auto pDataHandler = std::make_shared<VisionaryTMiniData>();
  VisionaryDataStream dataStream(pDataHandler);
  VisionaryControl visionaryControl;

  // 连接到相机数据流
  if (!connectToCamera(dataStream, deviceIpAddr, deviceBlobCtrlPort))
  {
    return 1; // 如果无法连接到相机，则退出程序
  }

  // 连接到相机控制通道
  if (!visionaryControl.open(VisionaryControl::ProtocolType::COLA_2, deviceIpAddr.c_str(), 5000 /*ms*/))
  {
    std::printf("Failed to open control connection to device.\n");
    return false; // 连接失败
  }

  // 读取设备标识
  std::printf("DeviceIdent: '%s'\n", visionaryControl.getDeviceIdent().c_str());
  visionaryControl.startAcquisition();
  #pragma endregion

  // 加载YOLO模型
  #pragma region load_model
  Yolov8Custom yolo;
  yolo.LoadModel(model_file);
  yolo.setStaticParams(0.1f, 0.8f, "top_1_labels_list.txt", 1);
  int ch = 0;
  #pragma endregion

  // MQTT连接配置
  #pragma region mqtt_connect_info
  mqtt::async_client cli(SERVER_ADDRESS, CLIENT_ID);
  mqtt::connect_options connOpts;
  connOpts.set_clean_session(false);

  // 安装回调函数
  callback cb(cli, connOpts);
  cli.set_callback(cb);
  try
  {
    std::cout << "Connecting to the MQTT server..." << std::flush;
    cli.connect(connOpts, nullptr, cb);
  }
  catch (const mqtt::exception &exc)
  {
    std::cerr << "\nERROR: Unable to connect to MQTT server: '"
              << SERVER_ADDRESS << "'" << exc << std::endl;
    return 1;
  }
  #pragma endregion

  // 主循环
  while (true)
  {
    // 获取下一帧数据
    if (!dataStream.getNextFrame())
    {
      continue; // 如果没有收到有效帧，继续下一次循环
    }

    // 相机检测模式
    if (CAMERA_RUN)
    {
      // 获取相机参数和图像数据
      int width = pDataHandler->getCameraParameters().width;
      int height = pDataHandler->getCameraParameters().height;
      std::vector<uint16_t> intensityMap = pDataHandler->getIntensityMap();
      std::vector<PointXYZ> pointCloud;
      pDataHandler->generatePointCloud(pointCloud);
      pDataHandler->transformPointCloud(pointCloud);
      std::cout << pointCloud.size() << std::endl;

      // 处理强度图
      cv::Mat image(height, width, CV_16U);
      for (int i = 0; i < height; ++i)
      {
        for (int j = 0; j < width; ++j)
        {
          image.at<uint16_t>(i, j) = static_cast<uint16_t>(intensityMap[i * width + j]);
        }
      }
      cv::Mat normalizedMap;
      cv::normalize(image, normalizedMap, 0, 255, cv::NORM_MINMAX, CV_8U);
      cv::Mat adjusted_image;
      image.convertTo(adjusted_image, -1, 0.3, 10);
      cv::imwrite("./adjusted_image.jpg", adjusted_image);
      cv::Mat img = cv::imread("./adjusted_image.jpg");

      // 处理距离图
      cv::Mat distanceImage(height, width, CV_16U);
      for (int i = 0; i < height; ++i)
      {
        for (int j = 0; j < width; ++j)
        {
          distanceImage.at<uint16_t>(i, j) = static_cast<uint16_t>(pointCloud[i * width + j].z * 1000);
        }
      }

      // 运行YOLO检测
      std::vector<Detection> objects2;
      cv::Mat normalizedDistanceMap;
      cv::normalize(distanceImage, normalizedDistanceMap, 0, 255, cv::NORM_MINMAX, CV_8U);
      yolo.RunObb(img, objects2);

      // 计算检测框的平均值
      std::vector<float16_t> averages;
      auto zValues = extractZValues(pointCloud, width, height);
      for (auto object : objects2)
      {
        float centerX = object.x;
        float centerY = object.y;
        float average = calculateSquareAverage(zValues, width, height, centerX, centerY, 3);
        cv::Point_<int> center(static_cast<int>(object.x), static_cast<int>(object.y));
        averages.push_back(average);
      }

      // 创建检测信息并转换为JSON
      InfoBox info(objects2.size(), averages);
      std::string payload = InfoBoxToJsonString(info);

      // 绘制检测结果并保存图像
      DrawDetectionsObb(img, objects2);
      cv::imwrite("./result_from_read.jpg", img);

      // 发布检测结果到MQTT
      const std::string topic = "camera/detector_num";
      std::cout << payload << std::endl;
      mqtt::message_ptr pubmsg = mqtt::make_message(topic, payload);
      pubmsg->set_qos(QOS);
      cli.publish(pubmsg)->wait();
      CAMERA_RUN = false;
    }

    // 抓取模式
    if (GRAB)
    {
      // 采集10张图像
      for (int i = 0; i < 10; i++)
      {
        int width = pDataHandler->getCameraParameters().width;
        int height = pDataHandler->getCameraParameters().height;
        std::vector<uint16_t> intensityMap = pDataHandler->getIntensityMap();
        std::vector<uint16_t> distanceMap = pDataHandler->getDistanceMap();

        // 生成随机图片名
        std::string jpgName = generateRandomImageName(".jpg");

        // 处理强度图
        cv::Mat image(height, width, CV_16U);
        for (int i = 0; i < height; ++i)
        {
          for (int j = 0; j < width; ++j)
          {
            image.at<uint16_t>(i, j) = static_cast<uint16_t>(intensityMap[i * width + j]);
          }
        }
        cv::Mat normalizedMap;
        cv::normalize(image, normalizedMap, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat adjusted_image;
        image.convertTo(adjusted_image, -1, 0.3, 10);
        cv::imwrite("./data/" + jpgName, adjusted_image);
      }
      GRAB = false;
    }

    // 高度测量模式
    if (LEFTING_HEIGHT)
    {
      // 检查数据处理器有效性
      if (!pDataHandler)
      {
        std::cerr << "Error: pDataHandler is null." << std::endl;
        return -1;
      }

      try
      {
        // 获取相机参数
        const auto &cameraParams = pDataHandler->getCameraParameters();
        if (cameraParams.width <= 0 || cameraParams.height <= 0)
        {
          std::cerr << "Error: Invalid camera parameters." << std::endl;
          return -1;
        }

        int width = cameraParams.width;
        int height = cameraParams.height;

        // 生成点云数据
        std::vector<PointXYZ> pointCloud;
        pDataHandler->generatePointCloud(pointCloud);
        if (pointCloud.empty())
        {
          std::cout << "No points in the point cloud." << std::endl;
          return 0;
        }

        // 设置ROI区域并过滤点云
        std::vector<PointXYZ> filteredPointCloud;
        for (const auto &point : pointCloud)
        {
          if (!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z) &&
              point.x >= -1.2 && point.x <= 1.0 &&
              point.y >= -1.1 && point.y <= 1.0)
          {
            filteredPointCloud.push_back(point);
          }
        }

        if (filteredPointCloud.empty())
        {
          std::cout << "No valid points found in the point cloud." << std::endl;
          return 0;
        }

        // 将点云分为16块进行处理
        int numBlocks = 16;
        if (numBlocks <= 0)
        {
          std::cerr << "Error: numBlocks must be positive." << std::endl;
          return -1;
        }

        if (width < numBlocks || height < numBlocks)
        {
          std::cerr << "Error: Image dimensions too small for the requested number of blocks." << std::endl;
          return -1;
        }

        // 计算每个块的平均值
        std::vector<float> blockAverages;
        const float xRange = 2.2f;
        const float yRange = 2.1f;

        for (int i = 0; i < numBlocks; ++i)
        {
          for (int j = 0; j < numBlocks; ++j)
          {
            std::vector<float> blockZValues;

            for (const auto &point : filteredPointCloud)
            {
              int xBlock = static_cast<int>(((point.x + 1.2f) / xRange) * numBlocks);
              int yBlock = static_cast<int>(((point.y + 1.1f) / yRange) * numBlocks);

              xBlock = std::max(0, std::min(numBlocks - 1, xBlock));
              yBlock = std::max(0, std::min(numBlocks - 1, yBlock));

              if (xBlock == i && yBlock == j)
              {
                blockZValues.push_back(point.z);
              }
            }

            if (!blockZValues.empty())
            {
              // 去除异常值
              int numPointsToRemove = static_cast<int>(std::ceil(blockZValues.size() * 0.1f));
              if (numPointsToRemove > 0 && blockZValues.size() > 2 * numPointsToRemove)
              {
                std::sort(blockZValues.begin(), blockZValues.end());
                blockZValues.erase(blockZValues.begin(), blockZValues.begin() + numPointsToRemove);
                blockZValues.erase(blockZValues.end() - numPointsToRemove, blockZValues.end());
              }

              if (!blockZValues.empty())
              {
                float blockAverage = std::accumulate(blockZValues.begin(), blockZValues.end(), 0.0f) / blockZValues.size();
                blockAverages.push_back(blockAverage);
              }
            }
          }
        }

        // 计算并发布最小高度值
        if (!blockAverages.empty())
        {
          float minAverage = *std::min_element(blockAverages.begin(), blockAverages.end());
          std::cout << "Minimum average Z value: " << minAverage << std::endl;

          const std::string topic = "camera/height";
          std::cout << "最高点(相对于相机的最近点)" << minAverage << std::endl;
          const std::string message = std::to_string(minAverage);
          mqtt::message_ptr pubmsg = mqtt::make_message(topic, message);
          pubmsg->set_qos(QOS);
          cli.publish(pubmsg)->wait();
        }
        else
        {
          std::cout << "No valid blocks found." << std::endl;
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return -1;
      }
      LEFTING_HEIGHT = false;
    }
  }

  // 断开MQTT连接
  try
  {
    std::cout << "\nDisconnecting from the MQTT server..." << std::flush;
    cli.disconnect()->wait();
    std::cout << "OK" << std::endl;
  }
  catch (const mqtt::exception &exc)
  {
    std::cerr << exc << std::endl;
    return 1;
  }

  // 关闭相机连接
  visionaryControl.close();
  dataStream.close();
}