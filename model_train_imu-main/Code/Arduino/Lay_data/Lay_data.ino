#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>

char *ssid = "PTIT.HCM_CanBo";     // Tên mạng WiFi của bạn
char *password = "";             // Mật khẩu mạng WiFi của bạn

const char *host = "10.251.13.222";     // Địa chỉ IP của máy tính chạy server
const int port = 8090;                 // Cổng mà server lắng nghe

const int numFeature = 6;              // Số lượng cột thuộc tính
int16_t numSample = 0;   //Đếm số lượng data thu được
const int maxNumSample = 800;  // Số dữ liệu lấy tối đa
const int timedelay = 25; //25ms lấy data 1 lần
boolean shouldStart = true; // Biến dừng thu data

MPU6050 mpu;
WiFiClient client;

// Mảng để lưu trữ dữ liệu thu được
float dataBuffer[maxNumSample][numFeature]; // Số lượng mẫu tối đa là 1000

void setup() {
  Wire.begin();
  Serial.begin(115200);

  mpu.initialize();

  //Kiểm tra kết nối MPU6050
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
    while (1);
  }

  // Kết nối WiFi
  WiFi.begin(ssid,password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Kết nối đến server
  if (client.connect(host, port)) {
    Serial.println("Connected to server");
  } else {
    Serial.println("Connection to server failed");
  }
  delay(3000);
}



void loop() {
  delay(timedelay);
  if (shouldStart) {
    float data[numFeature]; // Mảng để lưu dữ liệu

    // Thu thập dữ liệu từ cảm biến và lưu vào mảng
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Chuyển đổi dữ liệu thành các giá trị float và lưu vào mảng
    data[0] = ax / 16384.0;
    data[1] = ay / 16384.0;
    data[2] = az / 16384.0;
    data[3] = gx / 131.0;
    data[4] = gy / 131.0;
    data[5] = gz / 131.0;
    
    // Lưu dữ liệu mỗi lần thu đc vào mảng buffer
    for (int i = 0; i < numFeature; i++) {
      dataBuffer[numSample][i] = data[i];
      Serial.print(dataBuffer[numSample][i] );
      Serial.print(" ");
      
    }
    
    Serial.println();

    numSample++;

    // Nếu đã đủ mẫu, gửi dữ liệu lên server và đặt lại các biến
    if (numSample >= maxNumSample) {
      delay(1000);
      sendDataToServer();
      shouldStart = false;
    }
  }
  
}

void sendDataToServer() {
  Serial.println("Sending data to server...");
  String data_send = "";
  // Chuyển đổi data thành String gửi server
  for (int i = 0; i < maxNumSample; i++) {
    for (int j = 0; j < numFeature; j++) {
      data_send += String(dataBuffer[i][j]);
      if(j < numFeature -1){
        data_send += ",";
      }
    }

    if(i <  maxNumSample -1){
        data_send += "#";
      }
  }

  Serial.println("Data send : "+ data_send);
  // Gửi lên server 
  client.print(data_send);
  
  delay(3000);
  numSample = 0; // Đặt lại số lượng mẫu
  client.stop();
  Serial.println("End send data to server.");
}
