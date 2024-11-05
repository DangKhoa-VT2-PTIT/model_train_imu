#ifndef SENSORSETUP_H
#define SENSORSETUP_H
#include <MPU6050.h>
#include <Wire.h>

// Data collection params
const int num_timesteps = 60;
const int num_features = 6;
const int time_delay = 25;

// Init
MPU6050 mpu;

// array to save IMU data
float input_data[num_timesteps * num_features] = {};

void sensorSetup(void)
{
  Wire.begin();
  mpu.initialize();
  if (!mpu.testConnection()) 
  {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }
}

#endif