#include <TensorFlowLite_ESP32.h>
#include <MPU6050.h>
#include <Wire.h>

#include "secrets.h"
#include <Firebase.h>
#include <ArduinoJson.h>
#include <time.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "AImodel.h"
#include "FFT.h"

const int num_timesteps = 60;
const int num_features = 6;
const int time_delay = 25;

// Init
MPU6050 mpu;
Firebase fb(REFERENCE_URL);

// Timestamp define
// Cấu hình NTP server
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 7 * 3600;  // GMT+7 (có thể thay đổi tùy theo múi giờ của bạn)
const int   daylightOffset_sec = 0;
// unsigned int count = 0;

// ADC variable
int offset = 10;
int numberOfSamples = 1000;
int sum = 0;
int voltage = 0;
const int chargePin = 4;
const int fullPin = 5;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "Breathing",
  "Apnea"
};
String gestures = "";

// Số label dự đoán
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// Mảng dùng để lưu dữ liệu thu được và đưa vào model dự đoán 
float input_data[num_timesteps * num_features] = {};

int predicted_gesture = -1; // Biến lưu vị trí dự đoán
volatile bool data_ready = false;
bool gotData = false;
bool record = false;

void setup() {
  Wire.begin();
  Serial.begin(115200);
  WiFi.disconnect();
  delay(1000);

  /* Connect to WiFi */
  Serial.println();
  Serial.println();
  Serial.print("Connecting to: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print("-");
    delay(500);
  }

  Serial.println();
  Serial.println("WiFi Connected");
  Serial.println();

  // Khởi tạo NTP để đồng bộ thời gian
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  // Kiểm tra kết nối MPU6050
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(AImodel);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Create FreeRTOS tasks
  xTaskCreate(taskReadSensor, "TaskReadSensor", 4096, NULL, 1, NULL);
  xTaskCreate(taskProcessData, "TaskProcessData", 16384, NULL, 1, NULL);
  xTaskCreate(taskSendData, "TaskSendData", 8192, NULL, 1, NULL);
}

void taskReadSensor(void *parameter) {
  // Serial.print("taskReadSensor running on core ");
  // Serial.println(xPortGetCoreID());
  while (1) {
    // if (!fb.getBool("Record")) {
    //   vTaskDelay(1000 / portTICK_PERIOD_MS); // Delay trước khi kiểm tra lại
    //   continue; // Dừng gửi dữ liệu nếu record là false
    // }
    // Thu data
    for (int i = 0; i < num_timesteps; i++) {
      int16_t ax, ay, az;
      int16_t gx, gy, gz;
      mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

      // Chuyển đổi dữ liệu thành các giá trị float và lưu vào mảng
      input_data[i * num_features + 0] = ax / 16384.0;
      input_data[i * num_features + 1] = ay / 16384.0;
      input_data[i * num_features + 2] = az / 16384.0;
      input_data[i * num_features + 3] = gx / 131.0;
      input_data[i * num_features + 4] = gy / 131.0;
      input_data[i * num_features + 5] = gz / 131.0;
      delay(time_delay);
    }

    for (int i = 0; i < resp_samples; i++) 
    {
        float sum_val = 0;
        for(int j = 0; j < 4; j++)
        {
          int16_t ax, ay, az;
          mpu.getAcceleration(&ax, &ay, &az);
          sum_val += (float)az / 16384.0;
        }
        resp_data[i] = sum_val / 4;
        Serial.print(resp_data[i]);
        Serial.print(" ");
        delay(time_delay);
    }

    gotData = true;
    vTaskDelay(10 / portTICK_PERIOD_MS); // Delay to prevent this task from running too frequently
    yield();
  }
}

void taskProcessData(void *parameter) {
  // Serial.print("taskProcessData running on core ");
  // Serial.println(xPortGetCoreID());
  while (1) {
    // if (!fb.getBool("Record")) {
    //   vTaskDelay(1000 / portTICK_PERIOD_MS); // Delay trước khi kiểm tra lại
    //   continue; // Dừng gửi dữ liệu nếu record là false
    // }

    if(gotData){
      //Add data to model
      for (int i = 0; i < num_timesteps * num_features; i++) {
        tflInputTensor->data.f[i] = input_data[i];
      }

      // Run inferencing
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Invoke failed!");
        // while (1);
        return;
      }

      // Find the index of the maximum value in the output tensor
      int max_index = 0;
      float max_value = tflOutputTensor->data.f[0];
      for (int i = 1; i < NUM_GESTURES; i++) {
        if (tflOutputTensor->data.f[i] > max_value) {
          max_index = i;
          max_value = tflOutputTensor->data.f[i];
        }
      }
      // Cập nhật nhãn dự đoán
      predicted_gesture = max_index;

      // Dự đoán tư thế
      int16_t ax, ay, az;
      mpu.getAcceleration(&ax, &ay, &az);
      float ax_cal = ax / 16384.0 * 9.81;
      float ay_cal = ay / 16384.0 * 9.81;
      float az_cal = az / 16384.0 * 9.81;
      Serial.println(ax_cal);
      Serial.println(ay_cal);
      Serial.println(az_cal);

      if(-ay_cal >= 8)
      {
        gestures = "Sitting";
      }
      else if(az_cal >= 8)
      {
        gestures = "Lying";
      }

      // Tính Fourier Transform với dữ liệu từ các trục
      compute_fourier_transform(resp_data, frequencies, magnitudes, resp_samples);
      respiratory = rr_est(magnitudes, resp_samples);

      
      //Cập nhật trạng thái gotData
      gotData = false;
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); // Delay to prevent this task from running too frequently
    yield();
  }
}

void taskSendData(void *parameter) {
  // Serial.print("taskSenData running on core ");
  // Serial.println(xPortGetCoreID());
  const int sendInterval = 1000;  // Gửi data mỗi 1 giây (1000 milliseconds)
  unsigned long lastSendTime = 0; // Biến lưu trữ thời gian lần gửi trước

  while (1) {
    // if (!fb.getBool("Record")) {
    //   vTaskDelay(1000 / portTICK_PERIOD_MS); // Delay trước khi kiểm tra lại
    //   continue; // Dừng gửi dữ liệu nếu record là false
    // }
    unsigned long currentMillis = millis(); // Lấy thời gian hiện tại
    if (predicted_gesture != -1 && (currentMillis - lastSendTime >= sendInterval)) {
      lastSendTime = currentMillis;  // Cập nhật thời gian lần gửi gần nhất

      String postData = GESTURES[predicted_gesture];
      bool isBreathing = (predicted_gesture == 0);
      // Print out the data being sent
      Serial.println(postData);

      // dung lượng pin
      sum = 0;
      for(int i = 0; i < numberOfSamples; i++)
      {
        int analogVolts = analogReadMilliVolts(3) - offset;
        sum += analogVolts;
        delayMicroseconds(1);
      }
      voltage = sum / numberOfSamples;

      // Lấy thời gian hiện tại theo định dạng "YYYY-MM-DD HH:MM:SS"
      String currentTime = getFormattedTime();
      Serial.println("Current Time: " + currentTime);


      /* ----- Serialization: Set example data in Firebase ----- */

      // Tạo một JSON document để chứa dữ liệu đầu ra
      JsonDocument docOutput;

      // Thêm nhịp thở (float) và timestamp (chuỗi định dạng) vào JSON document
      if(isBreathing == false)
      {
        respiratory = 0;
      }
      else if(respiratory < 5 || respiratory > 30)
      {
        respiratory = 12;
      }
      docOutput["breathingRate"] = respiratory;  // nhịp thở
      docOutput["breathingStatus"] = isBreathing;  // trạng thái thở/ngưng thở
      docOutput["gestures"] = gestures; // tư thế
      docOutput["batteryVoltage"] = voltage * 2; // điện áp pin
      
      

      // // Chuỗi để chứa dữ liệu JSON đã serialize
      String output;

      // // Serialize JSON document sang chuỗi
      serializeJson(docOutput, output);

      // Sử dụng chính timestamp làm khóa để lưu trữ dữ liệu trên Firebase
      String firebasePath = "/BreathingData/" + currentTime;

      // Gửi dữ liệu đã serialize lên Firebase
      fb.setJson(firebasePath, output);

      // In ra Serial để kiểm tra
      Serial.println("Data sent to Firebase:");
      Serial.println(output);

      predicted_gesture = -1;
    }
    delay(1); // Delay for 1 second before next data sending
  }
  vTaskDelay(10 / portTICK_PERIOD_MS); 
  yield();
}


void loop() {
  // The loop is not needed as we use FreeRTOS tasks
  vTaskDelete(NULL); 
}

// Hàm để lấy thời gian định dạng "YYYY-MM-DD HH:MM:SS"
String getFormattedTime() 
{
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) 
  {
    Serial.println("Failed to obtain time");
    return "0000-00-00 00:00:00";  // Trả về giá trị mặc định nếu không lấy được thời gian
  }
  
  char timeString[20];  // Chuỗi để chứa thời gian định dạng
  strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(timeString);
}

