import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# CSV 파일 경로
csv_file = 'gesture_train_data.csv'

# CSV 파일을 데이터프레임으로 읽기
data = pd.read_csv(csv_file, header=None)

# 데이터와 라벨 분리
X = data.iloc[:, :-1].values  # 특징 데이터
y = data.iloc[:, -1].values   # 라벨 데이터


data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

# 라벨을 정수로 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터 전처리 (예시: 표준화)
X = (X - np.mean(X)) / np.std(X)

# 데이터 분할 (학습 및 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X.shape[1], 1), input_shape=(X.shape[1],)),  # 입력 데이터의 형태에 맞게 Reshape
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # 클래스 개수에 맞게 출력층 설정
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))