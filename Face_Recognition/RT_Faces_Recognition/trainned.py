import os
import cv2 as cv
import numpy as np

faces_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
leftEyes_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_lefteye_2splits.xml')
rightEyes_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_righteye_2splits.xml')
smile_haarCascade = cv.CascadeClassifier('haarcascades\haarcascade_smile.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = 'Images'

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_arr = cv.imread(img_path)

            img_gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            # faces_haarCascade
            faces_dtc = faces_haarCascade.detectMultiScale(img_gray, 1.6, 6)

            for (x, y, w, h) in faces_dtc:
                faces_roi = img_gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

            # leftEyes_haarCascade
            leftEyes_dtc = leftEyes_haarCascade.detectMultiScale(img_gray, 1.6, 6)

            for (lx, ly, lw, lh) in leftEyes_dtc:
                leftEyes_roi = img_gray[ly:ly + lh, lx:lx + lw]
                features.append(leftEyes_roi)
                labels.append(label)

            # rightEyes_haarCascade
            rightEyes_dtc = rightEyes_haarCascade.detectMultiScale(img_gray, 1.6, 6)

            for (rx, ry, rw, rh) in rightEyes_dtc:
                rightEyes_roi = img_gray[ry:ry + rh, rx:rx + rw]
                features.append(rightEyes_roi)
                labels.append(label)

            # smile_haarCascade
            smile_dtc = smile_haarCascade.detectMultiScale(img_gray, 1.6, 6)

            for (sx, sy, sw, sh) in smile_dtc:
                smile_roi = img_gray[sy:sy + sh, sx:sx + sw]
                features.append(smile_roi)
                labels.append(label)

create_train()
print('----- Training Done! ----')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.train(features, labels)
face_recognizer.save('RT_trained.yml')