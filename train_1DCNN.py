import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the dataset
# actions = ['left', 'right', 'front', 'back', 'stop','nothing']
# created_times = [1712312496, 1712315022, 1712321031]

# data = np.concatenate([
#     np.load(os.path.join('dataset', f'raw_{action}_{created_time}.npy'))
#     for action in actions
#     for created_time in created_times
# ], axis=0)

print(tf.__version__)
data = []

for file in os.listdir('dataset_cnn'):
    if file.startswith('raw_') and file.endswith('.npy'):
        data.append(np.load(os.path.join('dataset_cnn', file)))

data = np.concatenate(data, axis=0)

X = data[:, :-1]  # Features (angles)
y = data[:, -1]   # Labels

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for CNN input (add a channel dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the 1D CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))  

from focal_loss import BinaryFocalLoss
# Compile the model
model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics=['accuracy'])

# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

#history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
checkpoint = ModelCheckpoint('1DCNN_focalloss_best.keras', 
                             save_weights_only=False, 
                             save_best_only=True,  # Save only the best model
                             monitor='val_acc',  # Monitor validation accuracy
                             verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=25, verbose=1, mode='auto')

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr])


# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# history = model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_test, y_test),
#     epochs=200,
#     batch_size = 32,
#     callbacks=[
#         ModelCheckpoint('1DCNN_Sofmax_nonclass.keras', verbose=1, save_best_only=True, mode='auto'),
#         ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=25, verbose=1, mode='auto')
#     ]
# )

# Save the model
model.save('1DCNN_focalloss.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Plot training history (loss and accuracy)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
