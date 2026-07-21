import os
import sys
import time
import signal
import numpy as np
import cv2
import serial
import onnxruntime as ort

from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model_EfficientNetB0_20260720_175329.onnx')
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

LABELS = {0: 'bukan_daur_ulang', 1: 'daur_ulang'}
SERVO_OPEN = b'OPEN\n'
SERVO_CLOSE = b'CLOSE\n'

session = None
ser = None
running = True


def init_onnx():
    global session
    if not os.path.exists(MODEL_PATH):
        print(f'Model not found: {MODEL_PATH}')
        sys.exit(1)
    session = ort.InferenceSession(MODEL_PATH)
    print(f'ONNX model loaded: {MODEL_PATH}')


def init_serial():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f'Serial connected: {SERIAL_PORT}')
    except serial.SerialException as e:
        print(f'Serial error: {e}')
        sys.exit(1)


def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return None
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print('Failed to capture frame')
        return None
    return frame


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict(img):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: img})
    prob = outputs[0][0][0]
    label_id = 1 if prob >= CONFIDENCE_THRESHOLD else 0
    return label_id, prob


def send_servo(command):
    if ser and ser.is_open:
        ser.write(command)
        print(f'Sent: {command.strip().decode()}')


def signal_handler(sig, frame):
    global running
    print('\nShutting down...')
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)

    print('=== Waste Sorting AI ===')
    init_onnx()
    init_serial()

    send_servo(SERVO_CLOSE)
    print('Waiting for hand detection from Arduino...')

    while running:
        if ser and ser.in_waiting:
            line = ser.readline().decode().strip()
            if line == 'HAND_DETECTED':
                print('\nHand detected! Capturing image...')
                frame = capture_image()
                if frame is None:
                    continue

                cv2.imwrite('capture_last.jpg', frame)
                print('Image captured.')

                img = preprocess(frame)
                label_id, prob = predict(img)
                label = LABELS[label_id]
                confidence = prob if label_id == 1 else 1 - prob

                print(f'Prediction: {label} ({confidence:.2%})')

                if label_id == 1:
                    print('Recyclable -> Opening bin')
                    send_servo(SERVO_OPEN)
                    time.sleep(5)
                    send_servo(SERVO_CLOSE)
                else:
                    print('Not recyclable -> Bin stays closed')
                    send_servo(SERVO_CLOSE)

                print('Waiting for next detection...')

        time.sleep(0.05)

    if ser and ser.is_open:
        ser.close()
    print('Done.')


if __name__ == '__main__':
    main()
