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

distance_level = 0

def face_data(img, call_out, distance_level):
    face_width = 0
    face_x, face_y = 0, 0
    face_center_x, face_center_y = 0, 0
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in face_detector:
    #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     face_width = w
    # return face_width

        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_width = w
        face_center = []

        # Drwaing circle at the center of the face
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y

        if distance_level < 10:
            distance_level = 10

        if call_out == True:
            cv.circle(img, (face_center_x, face_center_y), 2, (0, 255, 0), 1)
    return face_width, face_detector, face_center_x, face_center_y

img = cv.imread('img_CBDE.jpg')
img_face_width, _, _, _ = face_data(img, False, distance_level)
img_focal_length = Focal_Length(FACE_DISTANCE, FACE_WIDTH, img_face_width)
print(f'Focal Length: {round(img_focal_length, 2)}')

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 540)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

while True:
    isTrue, frame = capture.read()

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, distance_level)
    for (face_x, face_y, face_w, face_h) in Faces:
        if face_width_in_frame != 0:
            Distance = Distance_Finder(img_focal_length, FACE_WIDTH, face_width_in_frame)
            Distance_level = int(Distance)

            cv.putText(frame, f"Distance {round(Distance, 2)} CM", 
                        (face_x - 6, face_y - 6), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()