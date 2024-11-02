//ML and DSP library
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "AImodel.h"
#include "FFT.h"

// Sensor library
#include <MPU6050.h>
#include <Wire.h>

// Network and JSON 
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <time.h>
#include "secrets.h"

// Data collection params
const int num_timesteps = 60;
const int num_features = 6;
const int time_delay = 25;

// Init
MPU6050 mpu;

// Timestamp define
// Config NTP server
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 7 * 3600;  // GMT+7
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

// Number of label
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// array to save IMU data
float input_data[num_timesteps * num_features] = {};

int predicted_gesture = -1; // prediction index
volatile bool data_ready = false;
bool gotData = false;
bool record = false;

void setup() 
{
  Serial.begin(115200);
  
  // MPU6050 setup
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) 
  {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }
  //Setup wifi
  Wifi_setup();

  // Setup NTP
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  // Setup MQTT
  MQTT_setup();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(AImodel);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) 
  {
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

void taskReadSensor(void *parameter) 
{
  // Serial.print("taskReadSensor running on core ");
  // Serial.println(xPortGetCoreID());
  while (1) 
  {
    // if (!fb.getBool("Record")) {
    //   vTaskDelay(1000 / portTICK_PERIOD_MS);
    //   continue;
    // }
    //Data collection
    for (int i = 0; i < num_timesteps; i++) 
    {
      int16_t ax, ay, az;
      int16_t gx, gy, gz;
      mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

      // Convert datatype to float and save to array
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
        delay(time_delay);
    }

    gotData = true;
    vTaskDelay(10 / portTICK_PERIOD_MS); // Delay to prevent this task from running too frequently
  }
}

void taskProcessData(void *parameter) 
{
  // Serial.print("taskProcessData running on core ");
  // Serial.println(xPortGetCoreID());
  while (1) 
  {
    // if (!fb.getBool("Record")) {
    //   vTaskDelay(1000 / portTICK_PERIOD_MS);
    //   continue;
    // }

    if(gotData)
    {
      //Add data to model
      for (int i = 0; i < num_timesteps * num_features; i++) 
      {
        tflInputTensor->data.f[i] = input_data[i];
      }

      // Run inferencing
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) 
      {
        Serial.println("Invoke failed!");
        // while (1);
        return;
      }

      // Find the index of the maximum value in the output tensor
      int max_index = 0;
      float max_value = tflOutputTensor->data.f[0];
      for (int i = 1; i < NUM_GESTURES; i++) 
      {
        if (tflOutputTensor->data.f[i] > max_value) 
        {
          max_index = i;
          max_value = tflOutputTensor->data.f[i];
        }
      }
      // Update predict label
      predicted_gesture = max_index;

      // Predict gesture
      int16_t ax, ay, az;
      mpu.getAcceleration(&ax, &ay, &az);
      float ax_cal = ax / 16384.0 * 9.81;
      float ay_cal = ay / 16384.0 * 9.81;
      float az_cal = az / 16384.0 * 9.81;

      if(-ay_cal >= 8)
      {
        gestures = "Sitting";
      }
      else if(az_cal >= 8)
      {
        gestures = "Lying";
      }

      // FFT for respiratory
      compute_fourier_transform(resp_data, frequencies, magnitudes, resp_samples);
      respiratory = rr_est(magnitudes, resp_samples);

      
      //Update gotData state
      gotData = false;
    }
    vTaskDelay(10 / portTICK_PERIOD_MS); // Delay to prevent this task from running too frequently
  }
}

void taskSendData(void *parameter) 
{
  // Serial.print("taskSenData running on core ");
  // Serial.println(xPortGetCoreID());
  const int sendInterval = 1000;  
  unsigned long lastSendTime = 0;

  while (1) 
  {
    // if (!fb.getBool("Record")) {
    //   vTaskDelay(1000 / portTICK_PERIOD_MS);
    //   continue;
    // }
    unsigned long currentMillis = millis(); // get current time
    if (predicted_gesture != -1 && (currentMillis - lastSendTime >= sendInterval)) 
    {
      lastSendTime = currentMillis;

      String postData = GESTURES[predicted_gesture];
      bool isBreathing = (predicted_gesture == 0);
      // Print out the data being sent
      Serial.println(postData);

      // Battery capacity
      sum = 0;
      for(int i = 0; i < numberOfSamples; i++)
      {
        int analogVolts = analogReadMilliVolts(3) - offset;
        sum += analogVolts;
        delayMicroseconds(1);
      }
      voltage = sum / numberOfSamples;

      // Get current time by format "YYYY-MM-DD HH:MM:SS"
      String currentTime = getFormattedTime();
      Serial.println("Current Time: " + currentTime);


      /* ----- Serialization: Set example data in Firebase ----- */

      // Create JSON document
      JsonDocument docOutput;

      // Add processed data to JSON document
      if(isBreathing == false)
      {
        respiratory = 0;
      }
      else if(respiratory < 5 || respiratory > 30)
      {
        respiratory = 12;
      }
      docOutput["timestamp"] = currentTime;
      docOutput["breathingRate"] = respiratory; 
      docOutput["breathingStatus"] = isBreathing; 
      docOutput["gestures"] = gestures; 
      docOutput["batteryVoltage"] = voltage * 2;

      //JSON string
      String output;
      serializeJson(docOutput, output);

      // Publish data to MQTT topic
      if (client.connected()) 
      {
        client.publish(mqttTopic, output.c_str());
      } 
      else 
      {
        Serial.println("MQTT not connected");
      }

      predicted_gesture = -1;
    }
    delay(1); // Delay for 1 second before next data sending
  }
  vTaskDelay(10 / portTICK_PERIOD_MS); 
}


void loop() 
{
  // The loop is not needed as we use FreeRTOS tasks
  vTaskDelete(NULL); 
}

// Function to get formatted time "YYYY-MM-DD HH:MM:SS"
String getFormattedTime() 
{
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) 
  {
    Serial.println("Failed to obtain time");
    return "0000-00-00 00:00:00";  // Return default value if time cannot be obtained
  }
  
  char timeString[20];  // String to contain formatted time
  strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(timeString);
}

