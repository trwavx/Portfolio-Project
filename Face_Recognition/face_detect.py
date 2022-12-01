import cv2 as cv
import torch
from facenet_pytorch import MTCNN

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 540)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

while cap.isOpened():
    isSuccess, frame = cap.read()
    
    if isSuccess:
        boxes, props, landmarks = mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                frame = cv.rectangle(frame, (bbox[0], bbox[1]), 
                                      (bbox[2], bbox[3]), (0, 255, 0), 6)


    cv.imshow('Face Detection', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv.destroyAllWindows()