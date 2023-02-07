############################################
# Action recognition app
# George Dechichi Barbar
# Traind folder in LOGS -> tensorboard --logdir=.
############################################

from ast import Not
from multiprocessing.connection import wait
from pickle import FALSE
import cv2
from cv2 import FlannBasedMatcher
import numpy as np 
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential #Sequential Neural Network
from keras.layers import LSTM, Dense #Temporal component to active detection/Full connected 
from keras.callbacks import TensorBoard
from keras import optimizers
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score #Evaluates what is true/false positive/negative

# Variables to save landmarks collection
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..\Folders\Extract_Data')
actionFolders = np.array(['Hello', 'Thanks', 'iLoveYou'])   # Folder names to save the actions landmarks
no_sequence = 30                                            # Number of group of data used
sequence_length = 30                                        # Number of frames by group used to detect the action
wait_time = 3
actionsNames = {label:num for num, label in enumerate(actionFolders)} # Actions descriptions

# Data Arrays
sequences, labels = [], []
for action in actionFolders:
    for sequence in range(no_sequence):
        window = []
        for frameNum in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(actionsNames[action])
X = np.array(sequences)
Y = to_categorical(labels).astype(int)

# Train Section - Neural Network
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..\Logs')
tbCallback = TensorBoard(log_dir = log_dir)

model = Sequential() # Initiates the model
model.add(LSTM(64, return_sequences=True, activation='relu')) # Add LSTM layer - Units, Return info, activation function, shape
model.add(LSTM(128, return_sequences=True, activation='relu')) # Add LSTM layer
model.add(LSTM(64, return_sequences=False, activation='relu')) # Add LSTM layer - do not return because next layer is Dense
model.add(Dense(64, activation='relu')) # Add Dense layer
model.add(Dense(32, activation='relu')) # Add Dense layer
model.add(Dense(actionFolders.shape[0], activation='softmax')) # Add Dense layer - return value from 0 to 1 (probability)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy']) # Optimizer, loss function to be used need to be this because it is a multiclass classification, metrics is optional but good trach accuracy

model.fit(X_train, Y_train, epochs=300, callbacks=[tbCallback])
#model.summary()

# Predictions related to the sign
res = model.predict(X_test)
model.save('action.hdf5')
#model.keras.models.load_model('action.h5')

# Evaluating
yhat = model.predict(X_train)

ytrue = np.argmax(Y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)
acc = accuracy_score(ytrue, yhat)

print(f"Model Accuracy {acc*100}%")