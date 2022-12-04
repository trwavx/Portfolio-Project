import cv2 as cv

FACE_WIDTH = 15
FACE_DISTANCE = 40

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def Focal_Length(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def Distance_Finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance

def face_data(img):
    face_width = 0
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in face_detector:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_width = w
    return face_width

img = cv.imread('img_CBDE.jpg')
img_face_width = face_data(img)
img_focal_length = Focal_Length(FACE_DISTANCE, FACE_WIDTH, img_face_width)
print(f'Focal Length: {round(img_focal_length, 2)}')

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 540)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

while True:
    isTrue, frame = capture.read()
    frame_face_width = face_data(frame)

    if frame_face_width != 0:
        distance = Distance_Finder(img_focal_length, FACE_WIDTH, frame_face_width)
        cv.putText(frame, f'Distance: {round(distance, 2)} CM', 
                    (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('CBDE', frame)
    
    if cv.waitKey(1) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()