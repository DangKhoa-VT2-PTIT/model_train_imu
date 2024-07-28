#include <TensorFlowLite_ESP32.h>
#include <MPU6050.h>
// #include <WiFi.h>
#include <Wire.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gesture_model.h"

// const char *ssid = "PTIT.HCM_SV";     // Tên mạng WiFi của bạn
// const char *password = "";             // Mật khẩu mạng WiFi của bạn

// const char *host = "10.252.14.243";     // Địa chỉ IP của máy tính chạy server
// const int port = 80;                 // Cổng mà server lắng nghe

// String url = "/nhung/post.php";
// String postVariable = "temp=";

const int num_timesteps = 60;
const int num_features = 6;
const int time_delay = 25;


MPU6050 mpu;
// WiFiClient client;

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
  "Not Breathing"
};

// Số label dự đoán
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// Mảng dùng để lưu dữ liệu thu được và đưa vào model dự đoán 
float input_data[num_timesteps * num_features] = {};
int predicted_gesture = -1; // Biến lưu vị trí dự đoán
bool gotData = false;

void setup() {
  Wire.begin();
  Serial.begin(115200);

  mpu.initialize();

  // Kiểm tra kết nối MPU6050
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }

  // Kết nối WiFi
  // WiFi.begin(ssid, password);
  // while (WiFi.status() != WL_CONNECTED) {
  //   delay(100);
  //   Serial.print(".");
  // }
  // Serial.println("\nWiFi connected");
  // Serial.print("IP address: ");
  // Serial.println(WiFi.localIP());

  // Kết nối đến server
  // if (client.connect(host, port)) {
  //   Serial.println("Connected to server");
  // } else {
  //   Serial.println("Connection to server failed");
  // }

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(gesture_model);
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
  xTaskCreatePinnedToCore(taskReadSensor, "TaskReadSensor", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskProcessData, "TaskProcessData", 8192, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(taskSendData, "TaskSendData", 4096, NULL, 1, NULL, 1);
}

void taskReadSensor(void *parameter) {
  while (1) {
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
    gotData = true;
    vTaskDelay(20 / portTICK_PERIOD_MS); // Delay to prevent this task from running too frequently
    yield();
  }
}

void taskProcessData(void *parameter) {
  while (1) {
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
      //Cập nhật trạng thái gotData
      gotData = false;
    }
    vTaskDelay(20 / portTICK_PERIOD_MS); // Delay to prevent this task from running too frequently
    yield();
  }
}

void taskSendData(void *parameter) {

  while (1) {
    if (predicted_gesture != -1) {
      // Serial.print("Data send server: ");
      // Serial.println(GESTURES[predicted_gesture]);
      // client.print(GESTURES[predicted_gesture]);
      // predicted_gesture = -1; // Reset predicted_gesture sau khi gửi
      // Kết nối đến server
      // if (client.connect(host, port)) {
      //   Serial.println("Connected to server");
      // } else {
      //   Serial.println("Connection to server failed");
      // }
      String postData = GESTURES[predicted_gesture];
      // Send HTTP POST request
      // client.print("POST " + url + " HTTP/1.1\r\n");
      // client.print("Host: ");
      // client.println(host); // host
      // client.println("Connection: close");
      // client.println("Content-Type: application/x-www-form-urlencoded");
      // client.print("Content-Length: ");
      // client.println(postData.length()); // Calculate length of data to send

      // Print out the data being sent
      Serial.println();
      Serial.print(postData);

      // Wait for server response
      // while (client.connected()) {
      //   if (client.available()) {
      //     String line = client.readStringUntil('\r');
      //     Serial.print(line);
      //   }
      // }

      // client.stop();
      predicted_gesture = -1;
    }
      delay(50); // Delay for 1 second before next data sending
    
  }
  vTaskDelay(20 / portTICK_PERIOD_MS); 
  yield();
}


void loop() {
  // The loop is not needed as we use FreeRTOS tasks
}
