import cv2 as cv
import numpy as np

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

FONT = cv.FONT_HERSHEY_COMPLEX

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

net = cv.dnn.readNet('dnn_model/yolov4-tiny.cfg', 'dnn_model/yolov4-tiny.weights')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []

with open('dnn_model/classes.txt', 'r') as file:
    for class_name in file.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

faces_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
leftEyes_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_lefteye_2splits.xml')
rightEyes_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_righteye_2splits.xml')
smile_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_smile.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('RT_trained.yml')

capture = cv.VideoCapture('Videos\BenAfflek4.mp4')

while True:
    isTrue, frame = capture.read()
    # Detect Objects
    class_ids, scores, bboxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (class_id, scores, bbox) in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        
        cv.putText(frame, class_name, (x, y - 10), FONT, 1, BLUE, 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Regcognize Face
    faces_dtc = faces_haarCascade.detectMultiScale(frame_gray, 1.6, 6)

    for (x, y, w, h) in faces_dtc:
        faces_roi = frame_gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)

        cv.putText(frame, str(people[label]), (x, y - 10), FONT, 1, GREEN, 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

    # leftEyes_haarCascade
    leftEyes_dtc = leftEyes_haarCascade.detectMultiScale(frame_gray, 1.6, 6)
    
    for (lx, ly, lw, lh) in leftEyes_dtc:
        leftEyes_roi = frame_gray[ly:ly + lh, lx:lx + lw]
        label, confidence = face_recognizer.predict(leftEyes_roi)
        
        cv.rectangle(frame, (lx, ly), (lx + lw, ly + lh), GREEN, 2)

    # rightEyes_haarCascade
    rightEyes_dtc = rightEyes_haarCascade.detectMultiScale(frame_gray, 1.6, 6)
    
    for (rx, ry, rw, rh) in rightEyes_dtc:
        rightEyes_roi = frame_gray[ry:ry + rh, rx:rx + rw]
        label, confidence = face_recognizer.predict(rightEyes_roi)

        cv.rectangle(frame, (rx, ry), (rx + rw, ry + rh), GREEN, 2)

    # smile_haarCascade
    smile_dtc = smile_haarCascade.detectMultiScale(frame_gray, 1.6, 6)

    for (sx, sy, sw, sh) in smile_dtc:
        smile_roi = frame_gray[sy:sy + sh, sx:sx + sw]
        label, confidence = face_recognizer.predict(smile_roi)

        cv.rectangle(frame, (sx, sy), (sx + sw, sy + sh), GREEN, 2)

    cv.imshow('Video Face Recognizer', frame)

    if cv.waitKey(60) & 0xFF==27:
        break
    
capture.release()
cv.destroyAllWindows()