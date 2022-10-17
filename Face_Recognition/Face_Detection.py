import cv2 as cv

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

FONT = cv.FONT_HERSHEY_COMPLEX

capture = cv.VideoCapture(0)

capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

faces_haarCascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    isTrue, frame = capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_dtc = faces_haarCascade.detectMultiScale(frame_gray, scaleFactor=1.6, minNeighbors=6)
    
    for (x, y, w, h) in faces_dtc:
        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
    
    cv.imshow('Face Detection w/ OpenCV', frame)
    
    if cv.waitKey(30) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()