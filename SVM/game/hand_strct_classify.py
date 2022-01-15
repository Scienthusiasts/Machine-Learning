# coding=utf-8
import cv2
import numpy as np
import mediapipe as mp

from deal_hand_frame import distance
import sys;sys.path.append('../')
from SVM import SVM

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

W,H = 640,480
keypoint = []
cnt = 0

img0 = cv2.imread('0.png')
img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')

cls = {0:img2, 1:img0, 2:img1}
clstext1 = {0:"Paper", 1:"Rock", 2:"Scissors"}
clstext2 = {0:"Scissors", 1:"Paper", 2:"Rock"}
# 读取SVM权重
weight0 = np.load('./W&b_0.npy')
W0, b0 = weight0[:-1], weight0[-1]
weight1 = np.load('./W&b_1.npy')
W1, b1 = weight1[:-1], weight1[-1]
weight2 = np.load('./W&b_2.npy')
W2, b2 = weight2[:-1], weight2[-1]

def eval(X):
  # 测试(三个模型)
  res0 = SVM.linear_eval(X, W0, b0).reshape(-1)>0
  res1 = SVM.linear_eval(X, W1, b1).reshape(-1)>0
  res2 = SVM.linear_eval(X, W2, b2).reshape(-1)>0

  if(res0 and res1):pred = 0
  elif(not res0 and res2):pred = 1
  elif(not res1 and not res2):pred = 2
  else:pred = -1
  return pred





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

        X = distance(keypoint.reshape(1,21,3)).reshape(1,-1)
        pred = eval(X)
        w, h,_ = cls[pred].shape
        image[:w, -h:, :] = cls[pred]
        
        cv2.putText(image, 'your:' + clstext1[pred], (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, 'computer:' + clstext2[pred], (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Flip the image horizontally for a selfie-view display.
    if cnt<100:
      cv2.imwrite('./visualize_hand/%d.png' % cnt, image)
      cnt+=1
      print(cnt)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()