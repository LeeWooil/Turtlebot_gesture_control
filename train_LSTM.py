import numpy as np
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

actions = ['left', 'right', 'front', 'back', 'stop','?']
# created_times = [1712312496, 1712315022, 1712321031]

# data = np.concatenate([
#     np.load(os.path.join('data', f'seq_{action}_{created_time}.npy'))
#     for action in actions
#     for created_time in created_times
# ], axis=0)

data = np.concatenate([
    np.load(os.path.join('dataset_lstm', file))
    for file in os.listdir('dataset_lstm')
    if file.startswith('seq_') and file.endswith('.npy')
], axis=0)



print(data.shape)
x_data = data[:, :, :-1]
labels = data[:, 0, -1]
print (data)
print(x_data.shape)
print(labels.shape)


from tensorflow.keras.utils import to_categorical

y_data = to_categorical(labels, num_classes=len(actions))
y_data.shape

from sklearn.model_selection import train_test_split

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten


model = Sequential([
    LSTM(64, activation='tanh', return_sequences = True, input_shape=x_train.shape[1:3]),
    LSTM(32, activation = 'tanh'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    #Dense(len(actions), activation='sigmoid'),
    Dense(len(actions), activation='softmax')
])
from focal_loss import BinaryFocalLoss
model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['acc'])
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
checkpoint = ModelCheckpoint('LSTM_focalloss_best.keras', 
                             save_weights_only=False, 
                             save_best_only=True,  # Save only the best model
                             monitor='val_acc',  # Monitor validation accuracy
                             verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=25, verbose=1, mode='auto')

# Train the model
history = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val),
                    callbacks=[checkpoint, reduce_lr])

model.save('LSTM_focalloss.keras')

model.summary()


# history = model.fit(
#     x_train,
#     y_train,
#     validation_data=(x_val, y_val),
#     epochs=50,
#     batch_size = 32,
#     callbacks=[
#         ModelCheckpoint('LSTM_sigmoid+softmax.keras', verbose=1, save_best_only=True, mode='auto'),
#         ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
#     ]
# )
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model

model = load_model('LSTM_Sofmax_nonclass_200.keras')

y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))