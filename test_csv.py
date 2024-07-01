import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 함수: 각도 계산
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

# 모델 로드
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# 손 모델 로드
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    # 좌우 반전
    img = cv2.flip(img, 1)
    # 색상 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 손 감지
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 각도 계산
            angles = calculate_angles(hand_landmarks)

            # 모델 입력 형식에 맞게 데이터 전처리
            input_data = np.expand_dims(angles, axis=0)

            # 모델로 예측
            prediction = model.predict(input_data)
            gesture_label = np.argmax(prediction)

            # 예측 결과 텍스트로 표시
            if gesture_label == 0:
                cv2.putText(img, "Right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture_label == 1:
                cv2.putText(img, "Left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 화면에 표시
    cv2.imshow('Hand Gesture Detection', img)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()