import cv2 as cv

# Distance constants
KNOWN_DISTANCE = 30
KNOW_WIDTH = 14.3

# Color for object detected
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Focal length function
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Distance finder function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

# Detect Face
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Face data function:
def face_data(img):
    face_width = 0
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        face_width = w

    return face_width

# Initialize Picture
ref_img = cv.imread('ref_img.jpg')
ref_img_face_width = face_data(ref_img)
focal_length_found = FocalLength(KNOWN_DISTANCE, KNOW_WIDTH, ref_img_face_width)
print(focal_length_found)
cv.imshow('Face', ref_img)

# Initialize Video
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    face_width_in_frame = face_data(frame)

    if face_width_in_frame != 0:
        distance_finder = Distance_finder(focal_length_found, KNOW_WIDTH, face_width_in_frame)
        cv.putText(frame, f'Distance: {distance_finder}', (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow('Face istance Estimate', frame)

    if cv.waitKey(30) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()