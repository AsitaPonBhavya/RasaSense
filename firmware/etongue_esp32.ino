// Rasa-Sense ESP32 firmware (reference design - not needed for the simulator demo)
// Reads the e-tongue sensor array and POSTs JSON to the dashboard's /api/ingest.
//
// Wiring (ESP32 ADC1 pins, usable while Wi-Fi is on):
//   GPIO34  pH probe module (PH-4502C) analog out
//   GPIO35  TDS / conductivity module analog out
//   GPIO32  Photodiode (BPW34 + LM358 TIA), LED at 450 nm through sample cuvette
//   GPIO33, 36, 39, 25*  Electrode array E1..E4 via INA333 instrumentation amps
//            (*use an ADS1115 over I2C for 4+ high-res channels in the full build)
//   GPIO4   DS18B20 temperature probe (OneWire)
#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASS";
const char* SERVER    = "http://192.168.1.10:5050/api/ingest";  // laptop running app.py

OneWire oneWire(4);
DallasTemperature tempSensor(&oneWire);

float avgVolts(int pin, int n = 20) {
  long sum = 0;
  for (int i = 0; i < n; i++) { sum += analogReadMilliVolts(pin); delay(2); }
  return sum / (float)n / 1000.0;
}

void setup() {
  Serial.begin(115200);
  tempSensor.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(300);
}

void loop() {
  tempSensor.requestTemperatures();
  float temp = tempSensor.getTempCByIndex(0);

  // Calibration constants come from 2-point (pH 4/7) and 1413 uS/cm standard calibration.
  float ph = 7.0 + (2.50 - avgVolts(34)) / 0.18;
  float v = avgVolts(35);
  float tds = (133.42 * v * v * v - 255.86 * v * v + 857.39 * v) * 0.5;
  float optical = -log10(max(avgVolts(32), 0.01f) / 2.80);   // absorbance vs blank reference
  float e1 = avgVolts(33) * 1000 / 10.0;                      // mV after INA gain of 10
  float e2 = avgVolts(36) * 1000 / 10.0;
  float e3 = avgVolts(39) * 1000 / 10.0;
  float e4 = avgVolts(25) * 1000 / 10.0;

  char body[200];
  snprintf(body, sizeof(body),
    "{\"ph\":%.2f,\"tds\":%.1f,\"optical\":%.3f,\"temp\":%.2f,\"e1\":%.1f,\"e2\":%.1f,\"e3\":%.1f,\"e4\":%.1f}",
    ph, tds, optical, temp, e1, e2, e3, e4);

  HTTPClient http;
  http.begin(SERVER);
  http.addHeader("Content-Type", "application/json");
  http.POST(body);
  http.end();
  Serial.println(body);
  delay(500);
}
