#ifndef SECRETS_H
#define SECRETS_H
/*

          WIFI SETUP

*/
// #define WIFI_SSID     "GalaxyM34"
// #define WIFI_PASSWORD "dangkhoa"
// #define WIFI_SSID     "Khuvuonthanhpho"
// #define WIFI_PASSWORD "kvttp0104"
#define WIFI_SSID     "PTIT.HCM_SV"
#define WIFI_PASSWORD ""

WiFiClientSecure espClient;
PubSubClient client(espClient);

// MQTT server info(default uncomment, comment for test)
// const char* mqttServer = "rabbitmq-001-pub.sa.wise-paas.com";;
// const int mqttPort = 1883;
// const char* mqttUser = "ed6e5a2a-5899-11ea-8729-f6bfce9fbbfd:f172c4b8-c811-4c76-a84e-4c6c78b43d8f";
// const char* mqttPassword = "3lAPmS96yZ3RBy5VNCF9y0Zux";

// MQTT server test(default comment, uncomment for test)
const char* mqttServer = "f34505a4b96445239183b394756a01e1.s1.eu.hivemq.cloud";;
const int mqttPort = 8883;
const char* mqttUser = "ndk_mqtt_demo";
const char* mqttPassword = "20112002Kh";

// MQTT topic
const char* mqttTopic = "breathing/data";
void Wifi_setup(void)
{
  WiFi.disconnect();
  delay(1000);

  /* Connect to WiFi */
  Serial.println();
  Serial.println();
  Serial.print("Connecting to: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print("-");
    delay(500);
  }

  Serial.println();
  Serial.println("WiFi Connected");
  Serial.println();
}

void MQTT_setup(void)
{
  // Set up MQTT connection
  client.setServer(mqttServer, mqttPort);
  espClient.setInsecure();

  // Connect to MQTT
  if (client.connect("ESP32Client", mqttUser, mqttPassword)) {
    Serial.println("MQTT Connected");
  } else {
    Serial.print("MQTT Connection failed, rc=");
    Serial.print(client.state());
  }
}
#endif

