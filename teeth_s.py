"""
Dental Robot: Dual-Camera Teeth and Enamel Caries Detection with Motion Control
Author: Omar Khaled Hassan
Date:14/06/2025
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
from time import time, sleep
from gpiozero import LEDBoard

from teeth_detection1 import (
    ObjectDetection,
    ToothTracker,
    ServoControllerPCA9685,
)

# -------------------- GPIO Setup --------------------
DC_MOTOR_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(DC_MOTOR_PIN, GPIO.OUT)
GPIO.output(DC_MOTOR_PIN, GPIO.LOW)

# -------------------- Camera Setup --------------------
teeth_camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
caries_camera = cv2.VideoCapture(2, cv2.CAP_V4L2)
for cam in [teeth_camera, caries_camera]:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

dual_camera_mode = caries_camera.isOpened()
if not dual_camera_mode:
    print("⚠️ Warning: Caries camera not found. Using teeth camera for both.")

# -------------------- Detection Setup --------------------
teeth_detector = ObjectDetection()
caries_detector = ObjectDetection()

teeth_tracker = ToothTracker("webcam_stand.npz")
caries_tracker = ToothTracker("micro.npz")

teeth_id = teeth_detector.get_id_by_class_name("teeths")
caries_id = caries_detector.get_id_by_class_name("enamel caries")

# -------------------- Servo Setup --------------------
servo = ServoControllerPCA9685(channel_x=0, channel_y=1)
servo.set_y_offset_bias(20)
servo.set_y_gain(0.08)

# -------------------- Stepper Setup --------------------
stepper_1 = LEDBoard("GPIO19", "GPIO26")  # X-axis
stepper_2 = LEDBoard("GPIO13", "GPIO16")  # Y-axis
stepper_3 = LEDBoard("GPIO20", "GPIO21")  # Z-axis

CW = 0
CCW = 1

def rotate_stepper(stepper, direction, steps, delay=0.00000005):
    for _ in range(steps):
        stepper.value = (direction, 0)
        sleep(delay)
        stepper.value = (direction, 1)

# -------------------- Logic Functions --------------------
def parse_pos(pos_text):
    if not pos_text or pos_text == "Not Detected":
        return None
    try:
        coords = [float(c.strip()) for c in pos_text.strip("()").split(",")]
        return coords if len(coords) == 3 else None
    except Exception as e:
        print(f"Error parsing position: {e}")
        return None

def create_output(parsed_pos):
    if not parsed_pos or len(parsed_pos) != 3:
        return None
    try:
        z = parsed_pos[2]
        is_caries = int(z >= 100)
        return [float(c) for c in parsed_pos] + [is_caries]
    except Exception as e:
        print(f"Error creating output: {e}")
        return None

# -------------------- Main Loop --------------------
output_history = []
last_caries_time = time()
detection_timeout = 30  # seconds

while True:
    start_time = time()

    ret1, teeth_frame = teeth_camera.read()
    ret2, caries_frame = caries_camera.read() if dual_camera_mode else (ret1, teeth_frame.copy())

    if not ret1 or not ret2:
        print("Camera read failed.")
        continue

    teeth_frame = cv2.flip(teeth_frame, 1)
    caries_frame = cv2.flip(caries_frame, 1)

    h, w, _ = caries_frame.shape
    center = (w // 2, h // 2)

    teeth_out, teeth_pos, _ = ToothTracker.process_frame(teeth_frame, teeth_detector, teeth_tracker, [teeth_id, caries_id], "Tooth")
    caries_out, caries_pos, offset = ToothTracker.process_frame(caries_frame, caries_detector, caries_tracker, [caries_id], "Caries", frame_center=center)

    if offset:
        last_caries_time = time()
        _, dy = offset
        if abs(dy) > 15:
            servo.set_angle_y(dy + servo.y_offset_bias)
    elif time() - last_caries_time > detection_timeout:
        print("No enamel for 30s — Resetting servo")
        servo.reset_y()

    pos_text = teeth_pos if teeth_pos != "Not Detected" else caries_pos
    parsed_pos = parse_pos(pos_text)
    output = create_output(parsed_pos)

    if output:
        print("Detected:", output)
        output_history.append(output)

        x, y, z, is_caries = list(map(int, output[:4]))

        # Motor Safety
        if z <= 150:
            GPIO.output(DC_MOTOR_PIN, GPIO.HIGH)
        else:
            GPIO.output(DC_MOTOR_PIN, GPIO.LOW)

        # Movement
        stepper_type = rotate_stepper
        steps = (lambda d: 40 if is_caries else 400)

        for axis_val, stepper, axis in zip([x, y, z], [stepper_1, stepper_2, stepper_3], "XYZ"):
            direction = CW if axis_val >= 0 else CCW
            for _ in range(abs(axis_val)):
                stepper_type(stepper, direction, steps(direction))
                print(f"Rotating {axis}: {abs(axis_val)} units {'CW' if direction == CW else 'CCW'}")

    # Display
    fps = 1.0 / (time() - start_time + 1e-6)
    cv2.putText(teeth_out, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Teeth Detection", teeth_out)
    cv2.imshow("Caries Detection", caries_out)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------- Cleanup --------------------
teeth_camera.release()
caries_camera.release()
servo.stop()
GPIO.cleanup()
cv2.destroyAllWindows()
