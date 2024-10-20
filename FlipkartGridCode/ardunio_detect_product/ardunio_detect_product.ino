#include <Stepper.h>

const int ir1Pin = 2;
const int ir2Pin = 4;
const int relayPin = 7;
const int stepsPerRevolution = 200;
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

boolean objectDetected = false;
boolean signalOneTriggered = false;
boolean signalTwoTriggered = false;
boolean motorStoped = false;
unsigned long startTime1 = 0;
unsigned long startTime2 = 0;
unsigned long endTime = 0;
float elapsedTime = 0;

void setup() {
  Serial.begin(9600);
  myStepper.setSpeed(60);  // Set the speed to 60 RPM
  pinMode(ir1Pin, INPUT);
  pinMode(ir2Pin, INPUT);
  pinMode(relayPin, OUTPUT);

  digitalWrite(relayPin, HIGH);
}

void loop() {
  int sensorOneValue = digitalRead(ir1Pin);
  int sensorTwoValue = digitalRead(ir2Pin);
  unsigned long now = millis();

  if (motorStoped == true) {
    String receivedData = Serial.readString();
    Serial.print(receivedData);

    if (receivedData == "Start") {
      digitalWrite(relayPin, HIGH);
      motorStoped = false;
    } 
    else if (receivedData == "Reject") {
      myStepper.step(stepsPerRevolution);  // Pushes the Product
      delay(1000);
    }
    
    goto skip;
  }

  if ((sensorOneValue == HIGH) && (objectDetected == true)) {
    objectDetected = false;
    endTime = millis();
    elapsedTime = endTime - startTime1;

    Serial.print(elapsedTime / 1000.0, 1);
    Serial.print(",");
    Serial.println("NOT TRIGGERED");
  }
  
  if ((sensorOneValue == LOW) && (objectDetected == false)) {
    startTime1 = millis();
    objectDetected = true;
    signalOneTriggered = true;
  }

  if ((sensorTwoValue == LOW) && (signalOneTriggered == true) && (objectDetected == false)) {
    startTime2 = millis();
    signalOneTriggered = false;
    signalTwoTriggered = true;
  }

  if ((signalTwoTriggered == true) && (startTime2 + (elapsedTime / 2) < now)) {
    signalTwoTriggered = false;
    Serial.println("TRIGGERED");
    digitalWrite(relayPin, LOW);
    motorStoped = true;
  }

  skip:
  delay(100);  // Small delay to debounce sensor reading
}
