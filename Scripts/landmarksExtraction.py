############################################
# Action recognition app
# George Dechichi Barbar
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

# Variables to save landmarks collection

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..\Folders\Extract_Data')
actionFolders = np.array(['Hello', 'Thanks', 'iLoveYou'])   # Folder names to save the actions landmarks
no_sequence = 30                                            # Number of group of data used
sequence_length = 30                                        # Number of frames by group used to detect the action
wait_time = 3
actionsNames = {label:num for num, label in enumerate(actionFolders)} # Actions descriptions

# Variables from mediapipe
mp_holistic = mp.solutions.holistic                         # Holistic module
mp_drawing = mp.solutions.drawing_utils                     # Drawing utilities

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

# Tries to create a new folder for each action
for action in actionFolders:
    for sequence in range(no_sequence):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Looping through the actions and sequences
bContinue = True
cap = cv2.VideoCapture(0)                                   # Access video camera/capture device (cap variable) - 0 should be webcam
# ... accessing mediapipe module
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actionFolders:
        for sequence in range(no_sequence):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()                             # Read the frame date from the capture devide - frame is actually the image

                # Detection and predictions
                image, results = mediapipe_detection(frame, holistic) 

                # Draw landmarks
                draw_landmarks(image, results)
                
                # Collection logic to wait between videos collections
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting Frames for {} | Video: {} | Frame: {}'.format(action, sequence, frame_num), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    for i in range(wait_time + 1):
                        nTime = wait_time - i
                        cv2.rectangle(image, (120, 210), (230, 240), (0,0,0), -1)
                        cv2.putText(image, 'Starting in {}'.format(nTime), (125,230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)                    # Show in the screen with the 'NAME' and the frame
                        cv2.waitKey(1000)
                else:
                    cv2.putText(image, 'Collecting Frames for {} | Video: {} | Frame: {}'.format(action, sequence, frame_num), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)                    # Show in the screen with the 'NAME' and the frame

                # Save the frame keypoints
                keypoints = extract_landmarks(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                if (os.path.exists(npy_path)):
                    os.remove(npy_path)
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):              # Break the imaging loop - Press 'q'
                    bContinue = False
                    break
            if not(bContinue):
                break
        if not(bContinue):
            break
            
    cap.release()                                           # Release capture device
    cv2.destroyAllWindows()                                 # Close all the windows

sequences, labels = [], []
for action in actionFolders:
    for sequence in range(no_sequence):
        window = []
        for frameNum in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frameNum)))
            window.append(res)
        sequences.append(window)
        labels.append(actionsNames[action])
