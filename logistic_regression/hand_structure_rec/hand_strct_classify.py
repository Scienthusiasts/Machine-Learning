# coding=utf-8
import cv2
import numpy as np
import mediapipe as mp

from deal_hand_frame import distance
import sys;sys.path.append('../')
from logistic import logistic

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

W,H = 640,480
keypoint = []
cnt = 0
model = logistic(mode="pretrain")


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoint = []
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                keypoint.append([x,y,z])
            keypoint = np.array(keypoint)

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        X = distance(keypoint.reshape(1,21,3)).reshape(-1)
        # print(int(model.eval(X)>0.5))
        cls = {0:'cloth', 1:"fist"}
        cv2.putText(image, cls[int(model.eval(X)>0.5)], (200,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        if cnt<100:
          cv2.imwrite('./visualize_hand/%d.png' % cnt, image)
          cnt+=1

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()