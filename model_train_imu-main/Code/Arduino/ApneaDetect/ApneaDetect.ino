//ML and DSP library
#include "TensorFlowLiteSetup.h"
#include "FFT.h"

// Sensor library
#include "SensorSetup.h"

// Network and JSON 
#include "NetworkSetup.h"
#include <ArduinoJson.h>

// Time stamp
#include <time.h>

// Battery
#include "BatterySetup.h"

// Timestamp define
// Config NTP server
const char* ntp_server = "pool.ntp.org";
const long  gmt_offset_sec = 7 * 3600;  // GMT+7
const int   daylight_offset_sec = 0;

// State 
bool gotData = false;
bool record = false;
String gestures = "";

void setup() 
{
  Serial.begin(115200);
  
  // MPU6050 setup
  sensorSetup();

  //Setup wifi
  wifiSetup();

  // Setup NTP
  configTime(gmt_offset_sec, daylight_offset_sec, ntp_server);

  // Setup MQTT
  mqttSetup();

  // Setup AI model
  tensorFlowLiteSetup();

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
      for (int i = 1; i < NUM_STATES; i++) 
      {
        if (tflOutputTensor->data.f[i] > max_value) 
        {
          max_index = i;
          max_value = tflOutputTensor->data.f[i];
        }
      }
      // Update predict label
      predicted_breathing_state = max_index;

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
    unsigned long currentMillis = millis(); // get current time
    if (predicted_breathing_state != -1 && (currentMillis - lastSendTime >= sendInterval)) 
    {
      lastSendTime = currentMillis;

      String postData = BREATHING_STATES[predicted_breathing_state];
      bool isBreathing = (predicted_breathing_state == 0);
      // Print out the data being sent
      Serial.println(postData);

      // Battery capacity
      sum = 0;
      for(int i = 0; i < number_of_samples; i++)
      {
        int analogVolts = analogReadMilliVolts(3) - voltage_offset;
        sum += analogVolts;
        delayMicroseconds(1);
      }
      voltage = sum / number_of_samples;

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

      predicted_breathing_state = -1;
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

