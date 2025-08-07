/*
 * 🤖 Autonomous Robotic Hand - Arduino Gripper Control
 * 
 * This Arduino sketch controls a servo-based gripper for the autonomous
 * robotic hand system. It receives commands via serial communication
 * from the Python application.
 */

#include <Servo.h>

// Pin definitions
const int GRIPPER_SERVO_PIN = 9;  // Servo pin for gripper control
const int LED_PIN = 13;           // Built-in LED for status indication

// Servo object
Servo gripperServo;

// Gripper states
const int GRIPPER_OPEN = 0;      // Open position
const int GRIPPER_CLOSED = 1;    // Closed position
const int GRIPPER_PARTIAL = 2;   // Partial grasp position

// Servo angle limits
const int SERVO_OPEN_ANGLE = 180;    // Fully open position
const int SERVO_CLOSED_ANGLE = 0;    // Fully closed position
const int SERVO_PARTIAL_ANGLE = 90;  // Partial grasp position

// Current state
int currentGripperState = GRIPPER_OPEN;
int currentServoAngle = SERVO_OPEN_ANGLE;

// Serial communication
String inputString = "";
bool stringComplete = false;

// Timing
unsigned long lastCommandTime = 0;
const unsigned long COMMAND_TIMEOUT = 5000; // 5 seconds

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize servo
  gripperServo.attach(GRIPPER_SERVO_PIN);
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  
  // Set initial position
  gripperServo.write(SERVO_OPEN_ANGLE);
  currentServoAngle = SERVO_OPEN_ANGLE;
  
  // Wait for servo to reach position
  delay(1000);
  
  // Send ready signal
  Serial.println("OK: Gripper initialized");
  
  // Blink LED to indicate ready
  blinkLED(3);
}

void loop() {
  // Check for serial commands
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  // Check for timeout
  if (millis() - lastCommandTime > COMMAND_TIMEOUT) {
    // No recent commands, maintain current position
    lastCommandTime = millis();
  }
  
  // Small delay to prevent overwhelming the serial buffer
  delay(10);
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

void processCommand(String command) {
  command.trim();
  
  // Update last command time
  lastCommandTime = millis();
  
  // Parse command
  if (command.startsWith("G")) {
    // Grasp command
    int force = 50; // Default force
    
    // Parse force parameter if provided
    if (command.indexOf(":") != -1) {
      String params = command.substring(command.indexOf(":") + 1);
      force = parseParameter(params, "force", 50);
    }
    
    graspObject(force);
    
  } else if (command.startsWith("R")) {
    // Release command
    releaseObject();
    
  } else if (command.startsWith("P")) {
    // Partial grasp command
    int position = 50; // Default position
    
    // Parse position parameter if provided
    if (command.indexOf(":") != -1) {
      String params = command.substring(command.indexOf(":") + 1);
      position = parseParameter(params, "position", 50);
    }
    
    partialGrasp(position);
    
  } else if (command.startsWith("S")) {
    // Status command
    sendStatus();
    
  } else if (command.startsWith("C")) {
    // Calibrate command
    calibrateGripper();
    
  } else {
    // Unknown command
    Serial.println("ERROR: Unknown command");
  }
}

void graspObject(int force) {
  // Map force (0-100) to servo angle
  int targetAngle = map(force, 0, 100, SERVO_OPEN_ANGLE, SERVO_CLOSED_ANGLE);
  
  // Move servo to grasp position
  moveServoTo(targetAngle);
  
  currentGripperState = GRIPPER_CLOSED;
  
  // Send confirmation
  Serial.print("OK: Grasped with force ");
  Serial.println(force);
  
  // Indicate with LED
  digitalWrite(LED_PIN, HIGH);
}

void releaseObject() {
  // Move servo to open position
  moveServoTo(SERVO_OPEN_ANGLE);
  
  currentGripperState = GRIPPER_OPEN;
  
  // Send confirmation
  Serial.println("OK: Released");
  
  // Indicate with LED
  digitalWrite(LED_PIN, LOW);
}

void partialGrasp(int position) {
  // Map position (0-100) to servo angle
  int targetAngle = map(position, 0, 100, SERVO_OPEN_ANGLE, SERVO_CLOSED_ANGLE);
  
  // Move servo to partial position
  moveServoTo(targetAngle);
  
  currentGripperState = GRIPPER_PARTIAL;
  
  // Send confirmation
  Serial.print("OK: Partial grasp at position ");
  Serial.println(position);
  
  // Indicate with LED
  digitalWrite(LED_PIN, HIGH);
  delay(100);
  digitalWrite(LED_PIN, LOW);
}

void calibrateGripper() {
  Serial.println("INFO: Starting calibration...");
  
  // Move to open position
  moveServoTo(SERVO_OPEN_ANGLE);
  delay(500);
  
  // Move to closed position
  moveServoTo(SERVO_CLOSED_ANGLE);
  delay(500);
  
  // Move back to open position
  moveServoTo(SERVO_OPEN_ANGLE);
  delay(500);
  
  currentGripperState = GRIPPER_OPEN;
  
  Serial.println("OK: Calibration complete");
  
  // Indicate completion with LED
  blinkLED(5);
}

void sendStatus() {
  Serial.print("STATUS: State=");
  Serial.print(currentGripperState);
  Serial.print(", Angle=");
  Serial.print(currentServoAngle);
  Serial.print(", Connected=1");
  Serial.println();
}

void moveServoTo(int targetAngle) {
  // Constrain angle to valid range
  targetAngle = constrain(targetAngle, SERVO_CLOSED_ANGLE, SERVO_OPEN_ANGLE);
  
  // Move servo smoothly
  int currentAngle = currentServoAngle;
  int step = (targetAngle > currentAngle) ? 1 : -1;
  
  while (currentAngle != targetAngle) {
    currentAngle += step;
    gripperServo.write(currentAngle);
    delay(15); // Smooth movement
  }
  
  currentServoAngle = targetAngle;
}

int parseParameter(String params, String paramName, int defaultValue) {
  // Parse parameter string like "force=50,position=30"
  int startIndex = params.indexOf(paramName + "=");
  if (startIndex != -1) {
    startIndex += paramName.length() + 1;
    int endIndex = params.indexOf(",", startIndex);
    if (endIndex == -1) {
      endIndex = params.length();
    }
    String valueStr = params.substring(startIndex, endIndex);
    return valueStr.toInt();
  }
  return defaultValue;
}

void blinkLED(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
} 