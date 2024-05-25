import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import load_model

# 모델 파일 경로 지정
cnn_model_path = 'Turtlebot_gesture_control\model\gesture_recognition_model_v2.h5'
#lstm_model_path = 'model\gesture_recognition_model.keras'
lstm_model_path = 'models\gesture_recognition_model_v3.keras'

model_cnn = load_model(cnn_model_path)
model_lstm = load_model(lstm_model_path)

actions = ['left', 'right', 'front', 'back', 'stop']
created_times = [1714062265]

data_1d = np.concatenate([
    np.load(os.path.join('Turtlebot_gesture_control\dataset', f'raw_{action}_{created_time}.npy'))
    for action in actions
    for created_time in created_times
], axis=0)

X_test_1d = data_1d[:, :-1]  # Features (angles)
y_test_1d = data_1d[:, -1]   # Labels

y_test_1d = to_categorical(y_test_1d)

data_rnn = np.concatenate([
    np.load(os.path.join('Turtlebot_gesture_control\dataset', f'seq_{action}_{created_time}.npy'))
    for action in actions
    for created_time in created_times
], axis=0)

X_test_rnn = data_rnn[:, :, :-1]
y_test_rnn = data_rnn[:, 0, -1]


# from sklearn.metrics import accuracy_score

# 모델의 예측
y_pred_cnn = model_cnn.predict(X_test_1d)
y_pred_lstm = model_lstm.predict(X_test_rnn)

# 예측을 원-핫 인코딩에서 원래 클래스 레이블로 변환
y_pred_cnn_labels = np.argmax(y_pred_cnn, axis=1)
y_pred_lstm_labels = np.argmax(y_pred_lstm, axis=1)

# 실제 레이블
y_test_1d_labels = np.argmax(y_test_1d, axis=1)
y_test_rnn_labels = y_test_rnn  # RNN 모델의 경우, 원-핫 인코딩이 필요 없을 수 있습니다

# # 정확도 계산
accuracy_cnn = accuracy_score(y_test_1d_labels, y_pred_cnn_labels)
accuracy_lstm = accuracy_score(y_test_rnn_labels, y_pred_lstm_labels)

# print(f"1D CNN 모델의 정확도: {accuracy_cnn:.4f}")
# print(f"LSTM 모델의 정확도: {accuracy_lstm:.4f}")
from sklearn.metrics import classification_report
print(f"1D CNN 모델의 정확도: {accuracy_cnn:.4f}")
print(f"LSTM 모델의 정확도: {accuracy_lstm:.4f}")
# 1D CNN 모델의 평가 보고서
print("1D CNN 모델의 평가 보고서:")
print(classification_report(np.argmax(y_test_1d, axis=1), y_pred_cnn_labels, target_names=actions))

# LSTM 모델의 평가 보고서
print("LSTM 모델의 평가 보고서:")
print(classification_report(y_test_rnn, y_pred_lstm_labels, target_names=actions))

from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd

# 각 모델에 대한 평가 보고서를 생성합니다.
report_cnn = classification_report(y_test_1d_labels, y_pred_cnn_labels, target_names=actions, output_dict=True)
report_lstm = classification_report(y_test_rnn, y_pred_lstm_labels, target_names=actions, output_dict=True)

# 평가 지표를 pandas DataFrame으로 변환합니다.
df_cnn = pd.DataFrame(report_cnn).T
df_lstm = pd.DataFrame(report_lstm).T

# 평가 지표 그래프를 그립니다.
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# 1D CNN 모델의 평가 지표
sns.barplot(x=df_cnn.index, y='f1-score', data=df_cnn, ax=axs[0])
axs[0].set_title('1D CNN model F1 score')
axs[0].set_ylabel('F1 score')

# LSTM 모델의 평가 지표
sns.barplot(x=df_lstm.index, y='f1-score', data=df_lstm, ax=axs[1])
axs[1].set_title('LSTM model F1 score')
axs[1].set_ylabel('F1 score')

plt.show()