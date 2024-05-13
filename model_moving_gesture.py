import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling1D, LSTM, Dense, Flatten

# 데이터셋
data = np.load('data\gesture_moving_dataset.npy')
# 데이터셋 크기 확인
print("Data shape:", data.shape)

X = data[:, :, :-1]  # 입력 데이터 (samples, timesteps, features)
y = data[:, :, -1]   # 라벨 데이터 (samples, timesteps)

# 모델 구성
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 100)),
    Dense(1)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')  # 예시로 평균 제곱 오차(Mean Squared Error) 사용

# 모델 학습
model.fit(X, y, epochs=10, batch_size=32)

# 모델 요약
model.summary()