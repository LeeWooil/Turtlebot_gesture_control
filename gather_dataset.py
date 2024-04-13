import cv2
import mediapipe as mp
import numpy as np
import time

# 데이터 수집을 위한 변수 초기화
start_time = time.time()
stop_duration = 60  # 1분
go_duration = 60    # 1분
stop_data_collected = 0
go_data_collected = 0

max_num_hands = 1

mp_hands = mp.solutions.hands  # 손 인식
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

file_path = 'data/gesture_dataset.csv'
mode = 'a'  # 파일 열기 모드: 'a'는 기존 파일에 추가

cap = cv2.VideoCapture(0)

def calculate_angles(hand_landmarks):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

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
    return angle

# CSV 파일 열기를 try-except 구문으로 감싸서 예외 처리
try:
    with open(file_path, mode):
        pass  # 파일 열기만 시도하고 아무것도 하지 않음
except FileNotFoundError:
    print("CSV 파일을 찾을 수 없습니다.")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 현재 수집 중인 동작과 해당 동작의 학습 시간 표시
    current_time = time.time()
    if current_time - start_time <= stop_duration:
        gesture = "Right"
        label = 0
        remaining_time = int(stop_duration - (current_time - start_time))
    elif current_time - start_time <= stop_duration + go_duration:
        gesture = "left"
        label = 1
        remaining_time = int(stop_duration + go_duration - (current_time - start_time))
    else:
        gesture = "Finished"
        remaining_time = 0

    # 텍스트 추가
    cv2.putText(img, f"Gesture: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Remaining Time: {remaining_time} seconds", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            angle = calculate_angles(res)

            # Append angle data with label (e.g., 'stop' or 'go')
              # You can change the label as per your requirement
            data = np.hstack((angle, label))

            # 파일에 데이터 추가
            with open(file_path, mode) as f:
                np.savetxt(f, [data], delimiter=',', fmt='%f')

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q') or gesture == "Finished":
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()