import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'Images'

haar_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_arr = cv.imread(img_path)
            if img_arr is None:
                continue 
                
            img_gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.6, minNeighbors=6)

            for (x,y,w,h) in faces_rect:
                faces_roi = img_gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('PT_trained.yml')