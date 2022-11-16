import cv2 as cv
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while capture.isOpened():
    isTrue, frame = capture.read()

    if isTrue:
        boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                frame_rec = cv.rectangle(frame, (bbox[0], bbox[1])
                                        , (bbox[2], bbox[3]), (0, 255, 0), 6)

    cv.imshow('Faces Detection', frame)

    if cv.waitKey(1) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()