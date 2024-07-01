#!/usr/bin/env python
import cv2
import mediapipe as mp
import time
import rclpy
from std_msgs.msg import String

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    # Initialize ROS node
    rclpy.init()
    node = rclpy.create_node('hand_tracking_node')
    pub = node.create_publisher(String, 'status_hand', 10)

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
                        status(pub, "Front")
                        print("front")
                    elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
                        status(pub, "Back")
                        print("back")
                    elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
                        status(pub, "Left")
                    elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
                        status(pub, "Right")
                    else:
                        print("s")        

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            cv2.imshow('hand control', image)
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

