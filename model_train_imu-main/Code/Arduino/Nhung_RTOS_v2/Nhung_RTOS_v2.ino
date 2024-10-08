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

#include "gesture_model.h"

const int resp_samples = 400;
const int num_timesteps = 60;
const int num_features = 6;
const int time_delay = 25;

MPU6050 mpu;
Firebase fb(REFERENCE_URL);
// Firebase fb(REFERENCE_URL, AUTH_TOKEN);

const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 7 * 3600; 
const int daylightOffset_sec = 0;

int offset = 10;
int numberOfSamples = 1000;
int sum = 0;
int voltage = 0;
const int chargePin = 4;
const int fullPin = 5;

tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = { "Breathing", "Apnea" };
String gestures = "";
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

float input_data[num_timesteps * num_features] = {};
float resp_data[resp_samples * (num_features / 2)] = {};
float concatenated_data[resp_samples] = {};
int predicted_gesture = -1; 
float respiratory = 0; 
bool gotData = false;
bool record = false;

void setup() {
    Wire.begin();
    Serial.begin(115200);
    WiFi.disconnect();
    delay(1000);

    Serial.print("Connecting to: ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
        Serial.print("-");
        delay(500);
    }

    Serial.println("WiFi Connected");

    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while (1);
    }

    tflModel = tflite::GetModel(gesture_model);
    if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
    tflInterpreter->AllocateTensors();
    tflInputTensor = tflInterpreter->input(0);
    tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
    // Kiểm tra trạng thái ghi
    // if (!fb.getBool("Record")) {
    //     delay(1000);
    //     return; 
    // }

    // Đọc dữ liệu từ MPU6050
    for (int i = 0; i < num_timesteps; i++) {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

        // input_data[i * num_features + 0] = ax / 16384.0;
        // input_data[i * num_features + 1] = ay / 16384.0;
        // input_data[i * num_features + 2] = az / 16384.0;
        // input_data[i * num_features + 3] = gx / 131.0;
        // input_data[i * num_features + 4] = gy / 131.0;
        // input_data[i * num_features + 5] = gz / 131.0;

        // Chuẩn hóa gia tốc về khoảng -1 đến 1
        input_data[i * num_features + 0] = (ax / 16384.0) / 2.0;
        input_data[i * num_features + 1] = (ay / 16384.0) / 2.0;
        input_data[i * num_features + 2] = (az / 16384.0) / 2.0;

        // Chuẩn hóa con quay hồi chuyển về khoảng -1 đến 1
        input_data[i * num_features + 3] = (gx / 131.0) / 250.0;
        input_data[i * num_features + 4] = (gy / 131.0) / 250.0;
        input_data[i * num_features + 5] = (gz / 131.0) / 250.0;
        delay(time_delay);
    }

    

    for (int i = 0; i < resp_samples; i++) {
        int16_t ax, ay, az;
        mpu.getAcceleration(&ax, &ay, &az);

        resp_data[i * (num_features/2)] = (float)ax / 16384.0; 
        resp_data[i * (num_features/2) + 1] = (float)ay / 16384.0;
        resp_data[i * (num_features/2) + 2] = (float)az / 16384.0;
        delay(time_delay);
    }

    gotData = true;

    if (gotData) {
        for (int i = 0; i < num_timesteps * num_features; i++) {
            tflInputTensor->data.f[i] = input_data[i];
        }

        if (tflInterpreter->Invoke() != kTfLiteOk) {
            Serial.println("Invoke failed!");
            return;
        }

        int max_index = 0;
        float max_value = tflOutputTensor->data.f[0];
        for (int i = 1; i < NUM_GESTURES; i++) {
            if (tflOutputTensor->data.f[i] > max_value) {
                max_index = i;
                max_value = tflOutputTensor->data.f[i];
            }
        }

        predicted_gesture = max_index;

        int16_t ax, ay, az;
        mpu.getAcceleration(&ax, &ay, &az);
        float ax_cal = ax / 16384.0 * 9.81;
        float ay_cal = ay / 16384.0 * 9.81;
        float az_cal = az / 16384.0 * 9.81;

        if (-ay_cal >= 8) {
            gestures = "Sitting";
        } else if (az_cal >= 8) {
            gestures = "Lying";
        }

        remove_dc_and_concatenate(resp_data, concatenated_data);

        // float frequencies[resp_samples];
        // float magnitudes[resp_samples];
        // compute_fourier_transform(concatenated_data, frequencies, magnitudes, resp_samples);
        // respiratory = rr_est(magnitudes, resp_samples);

        gotData = false;
    }

    // Gửi dữ liệu lên Firebase
    if (predicted_gesture != -1) {
        String postData = GESTURES[predicted_gesture];
        bool isBreathing = (predicted_gesture == 0);
        
        sum = 0;
        for (int i = 0; i < numberOfSamples; i++) {
            int analogVolts = analogReadMilliVolts(3) - offset;
            sum += analogVolts;
            delayMicroseconds(1);
        }
        voltage = sum / numberOfSamples;

        String currentTime = getFormattedTime();

        JsonDocument docOutput;
        if (!isBreathing) {
            respiratory = 0;
        } else if (respiratory < 5 || respiratory > 30) {
            respiratory = 12;
        }
        docOutput["breathingRate"] = respiratory;
        docOutput["breathingStatus"] = isBreathing;
        docOutput["gestures"] = gestures;
        docOutput["batteryVoltage"] = voltage * 2;

        String output;
        serializeJson(docOutput, output);
        String firebasePath = "/BreathingData/" + currentTime;
        fb.setJson(firebasePath, output);
        Serial.println("Data sent to Firebase:");
        Serial.println(output);
        
        predicted_gesture = -1;
    }

    delay(1000); // Delay trước khi lặp lại
}

String getFormattedTime() {
    struct tm timeinfo;
    if (!getLocalTime(&timeinfo)) {
        Serial.println("Failed to obtain time");
        return "0000-00-00 00:00:00";
    }

    char timeString[20];
    strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S", &timeinfo);
    return String(timeString);
}

void remove_dc_and_concatenate(float* data, float* concatenated_data) {
    float mean_ax = 0, mean_ay = 0, mean_az = 0;

    for (int i = 0; i < resp_samples; i++) {
        mean_ax += data[i * (num_features/2)];
        mean_ay += data[i * (num_features/2) + 1];
        mean_az += data[i * (num_features/2) + 2];
    }
    mean_ax /= resp_samples;
    mean_ay /= resp_samples;
    mean_az /= resp_samples;

    // Loại bỏ thành phần DC và nối ax, ay, az thành một mảng
  for (int i = 0; i < resp_samples; i++) {
    concatenated_data[i] = (data[i * (num_features/2)] - mean_ax) +
                           (data[i * (num_features/2) + 1] - mean_ay) +
                           (data[i * (num_features/2) + 2] - mean_az);
    // Serial.print(concatenated_data[i]);
  }
}

//Tính Fourier Transform
void compute_fourier_transform(float* data, float* frequencies, float* magnitudes, int data_length) {
  Serial.println("Computing Fourier Transform...");

  // Loại bỏ thành phần DC
  float mean_value = 0;
  for (int i = 0; i < data_length; i++) {
    mean_value += data[i];
  }
  mean_value /= data_length;

  for (int i = 0; i < data_length; i++) {
    data[i] -= mean_value;
  }

  // Tính toán phần thực và phần ảo
  float real_part[data_length];
  float imag_part[data_length];

  for (int k = 0; k < data_length; k++) {
    real_part[k] = 0;
    imag_part[k] = 0;
    for (int n = 0; n < data_length; n++) {
      real_part[k] += data[n] * cos(2 * PI * n * k / data_length);
      imag_part[k] -= data[n] * sin(2 * PI * n * k / data_length);
    }
    magnitudes[k] = sqrt(real_part[k] * real_part[k] + imag_part[k] * imag_part[k]);
    frequencies[k] = k * 40.0 / data_length; // Tần số
  }

  Serial.println("Fourier Transform computed.");
}

// Tính resp
int rr_est(float* magnitudes, int samples) {
  Serial.println("Estimating breath rate from magnitudes...");
  int max_peak_index = 0;
  float max_peak_value = 0;

  for (int i = 1; i < samples - 1; i++) {
    if (magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1]) {
      if (magnitudes[i] > max_peak_value) {
        max_peak_value = magnitudes[i];
        max_peak_index = i;
      }
    }
  }

  // Chuyển đổi chỉ số đỉnh thành nhịp thở
  float breath_rate = (max_peak_index * 40.0 / samples) * 60; // Nhịp thở tính theo phút
  Serial.print("Nhịp thở ước lượng: ");
  Serial.println(breath_rate);

  return breath_rate;
}
