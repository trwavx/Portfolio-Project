import cv2 as cv
import torch
from facenet_pytorch import MTCNN
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(margin=20, keep_all=True, select_largest=True
                , post_process=False, device=device)

IMG_PATH = 'Faces_Dataset\Faces_Lists'
count = 50
usr_name = input('Enter your name: ')
USR_PATH = os.path.join(IMG_PATH, usr_name)
leap = 1

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while capture.isOpened() and count:
    isTrue, frame = capture.read()

    if mtcnn(frame) is not None and leap % 2:
        path = str(USR_PATH + '/{}.jpg'.format(str(count)))
        faces_img = mtcnn(frame, save_path=path)
        count -= 1
    leap += 1

    cv.imshow('Faces Capture', frame)

    if cv.waitKey(1) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()