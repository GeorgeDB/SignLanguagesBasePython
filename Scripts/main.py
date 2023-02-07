############################################
# Action recognition app
# George Dechichi Barbar
############################################

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
from tensorflow import keras

# Variables from mediapipe
mp_holistic = mp.solutions.holistic                         # Holistic module
mp_drawing = mp.solutions.drawing_utils                     # Drawing utilities

# Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.95

# Possible actions
actions = np.array(['Hello', 'Thanks', 'iLoveYou']) 

# Function to detect main points from the frame -> Get image - color conversion - determine points
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          # Color conversion
    image.flags.writeable = False                           # Image no longer writeable
    results = model.process(image)                          # Make position prediction
    image.flags.writeable = True                            # Image is writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)          # Color conversion back
    return image, results

# Function to draw the landmarks with formatting
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2),)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),)

# Function to pass the landmarks positions to arrays
def extract_landmarks(results):
# ... flatten arrays (needed in LSTM) with landmarks positions
    poseArr = (np.array([[res.x, res.y, res.z, res.visibility]  
        for res in results.pose_landmarks.landmark]).flatten()  
        if results.pose_landmarks else np.zeros(132))           # Zeros arrays for the case if there is no data in results (length = # of landmarks)
    faceArr = (np.array([[res.x, res.y, res.z]  
        for res in results.face_landmarks.landmark]).flatten()  
        if results.face_landmarks else np.zeros(1404))
    leftHandArr = (np.array([[res.x, res.y, res.z]  
        for res in results.left_hand_landmarks.landmark]).flatten()  
        if results.left_hand_landmarks else np.zeros(63))
    rightHandArr = (np.array([[res.x, res.y, res.z]  
        for res in results.right_hand_landmarks.landmark]).flatten()  
        if results.right_hand_landmarks else np.zeros(63))
    return np.concatenate([poseArr, faceArr, leftHandArr, rightHandArr])

# Rendering probabilities
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_action(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num*40), (int(prob * 100), 90 + num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_frame

# Imports trained model
model = keras.models.load_model('action.hdf5')

# Looping trhough frames
cap = cv2.VideoCapture(0)                                   # Access video camera/capture device (cap variable) - 0 should be webcam                             
# ... accessing mediapipe module
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():                                   # Check if the camera is opened
        ret, frame = cap.read()                             # Read the frame date from the capture devide - frame is actually the image

        # Detection and predictions
        image, results = mediapipe_detection(frame, holistic) 

        # Draw landmarks
        #draw_landmarks(image, results)

        # Prediction Logic
        keypoints = extract_landmarks(results)
        #sequence.insert(0,keypoints)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

        # Visualization logic
        try:
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
        except:
            continue

        if len(sentence) > 5:
            sentence = sentence[-5:]
        
        #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        #cv2.putText(image, ' '.join(sentence), (3, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Probability vizualization 
        image = prob_action(res, actions, image, colors)

        cv2.imshow('OpenCV Feed', image)                    # Show in the screen with the 'NAME' and the frame
        if cv2.waitKey(10) & 0xFF == ord('q'):              # Break the imaging loop - Press 'q'
            break
    cap.release()                                           # Release capture device
    cv2.destroyAllWindows()                                 # Close all the windows

