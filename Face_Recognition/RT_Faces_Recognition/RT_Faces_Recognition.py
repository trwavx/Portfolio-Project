import cv2 as cv
import numpy as np

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

FONT = cv.FONT_HERSHEY_COMPLEX

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
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # leftEyes_haarCascade
    leftEyes_dtc = leftEyes_haarCascade.detectMultiScale(frame_gray, 1.6, 6)
    
    for (lx, ly, lw, lh) in leftEyes_dtc:
        leftEyes_roi = frame_gray[ly:ly + lh, lx:lx + lw]
        label, confidence = face_recognizer.predict(leftEyes_roi)

    # rightEyes_haarCascade
    rightEyes_dtc = rightEyes_haarCascade.detectMultiScale(frame_gray, 1.6, 6)
    
    for (rx, ry, rw, rh) in rightEyes_dtc:
        rightEyes_roi = frame_gray[ry:ry + rh, rx:rx + rw]
        label, confidence = face_recognizer.predict(rightEyes_roi)

    # smile_haarCascade
    smile_dtc = smile_haarCascade.detectMultiScale(frame_gray, 1.6, 6)

    for (sx, sy, sw, sh) in smile_dtc:
        smile_roi = frame_gray[sy:sy + sh, sx:sx + sw]
        label, confidence = face_recognizer.predict(smile_roi)

    # faces_haarCascade
    faces_dtc = faces_haarCascade.detectMultiScale(frame_gray, 1.6, 6)

    for (x, y, w, h) in faces_dtc:
        faces_roi = frame_gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(faces_roi)

        cv.putText(frame, str(people[label]), (x, y - 10), FONT, 1, GREEN, 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)

    cv.imshow('Video Face Recognizer', frame)

    if cv.waitKey(60) & 0xFF==27:
        break
    
capture.release()
cv.destroyAllWindows()