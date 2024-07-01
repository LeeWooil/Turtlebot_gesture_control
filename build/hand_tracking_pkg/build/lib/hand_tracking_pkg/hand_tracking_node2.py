#!/usr/bin/env python
import cv2
import mediapipe as mp
import time
import rclpy
import numpy as np
import tensorflow as tf
from std_msgs.msg import String

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def calculate_angles(hand_landmarks):
    joint = np.zeros((21, 4))
    for j, lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

    # Compute angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
    v = v2 - v1  # [20,3]
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

    angle = np.degrees(angle)  # Convert radian to degree

    data = np.concatenate([joint.flatten(), angle])
            
    input_data = np.expand_dims(np.array(data, dtype=np.float32), axis=0)
    return input_data

def main():
    # Initialize ROS node
    rclpy.init()
    node = rclpy.create_node('hand_tracking_node')
    pub = node.create_publisher(String, 'status_hand', 10)

    model = tf.keras.models.load_model('Turtlebot_gesture_control/1DCNN_Sigmoid+Sofmax.h5')

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    input_data = calculate_angles(hand_landmarks)

                    prediction = model.predict(input_data)
                    gesture_label = np.argmax(prediction)
                    print(prediction)
                    max_pred1 = max(prediction)
                    max_pred = max(max_pred1)
                    print(max_pred)

                    if max_pred > 0.5:
                        actions = ['Left', 'Right', 'Front', 'Back', 'Stop']
                        gesture_text = actions[gesture_label]
                        status(pub,gesture_text)
                        cv2.putText(img, gesture_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        cv2.putText(img, "?", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            cv2.imshow('hand control', img)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

def status(pub, input):
    msg = String()
    msg.data = input
    pub.publish(msg)

if __name__ == '__main__':
    main()

