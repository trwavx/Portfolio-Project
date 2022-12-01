import cv2 as cv
import torch
from facenet_pytorch import MTCNN
from datetime import datetime
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IMG_PATH = 'faces_dataset/faces_list'
count = 50
usr_name = input("Enter your name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
leap = 1

mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, 
              post_process=False, device=device)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 540)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    
    if mtcnn(frame) is not None and leap % 2:
        path = str(USR_PATH + '/{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-")
                                               .replace(" ", "-") + str(count)))
        face_img = mtcnn(frame, save_path=path)
        count-=1
    leap+=1
    
    cv.imshow('Face Capturing', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv.destroyAllWindows()