import cv2 as cv
import numpy as np

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

FONT = cv.FONT_HERSHEY_COMPLEX

faces_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('PT_trained.yml')

img = cv.imread(r'Images\Ben Afflek\2.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_dtc = faces_haarCascade.detectMultiScale(img_gray, 1.6, 6)

for (x, y, w, h) in faces_dtc:
    faces_roi = img_gray[y:y + h, x:x + w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (x, y - 10), FONT, 1, GREEN, 2)
    cv.rectangle(img, (x,y), (x + w, y + h), GREEN, 2)

cv.imshow('Photo Face Recognizer', img)
cv.waitKey(0)