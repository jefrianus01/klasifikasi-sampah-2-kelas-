#include <Servo.h>

#define TRIG_PIN 9
#define ECHO_PIN 10
#define SERVO_PIN 6
#define HAND_DISTANCE_CM 15
#define TRIGGER_COOLDOWN_MS 3000

Servo servo;
unsigned long lastTriggerTime = 0;

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  servo.attach(SERVO_PIN);
  servo.write(0);
}

void loop() {
  if (millis() - lastTriggerTime < TRIGGER_COOLDOWN_MS) {
    return;
  }

  long distance = readUltrasonic();

  if (distance > 0 && distance <= HAND_DISTANCE_CM) {
    Serial.println("HAND_DETECTED");
    lastTriggerTime = millis();

    unsigned long timeout = millis() + 10000;
    while (millis() < timeout) {
      if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "OPEN") {
          servo.write(90);
        } else if (cmd == "CLOSE") {
          servo.write(0);
        }
      }
    }
    servo.write(0);
  }

  delay(100);
}

long readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  return duration * 0.034 / 2;
}
