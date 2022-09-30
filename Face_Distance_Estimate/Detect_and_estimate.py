import cv2 as cv

# Distance constant
KNOWN_DISTANCE = 40
KNOWN_WIDTH = 15

# Threshold constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors & Font
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

FONT = cv.FONT_HERSHEY_COMPLEX

# OpenCV dnn
net = cv.dnn.readNet('dnn_model\yolov4-tiny.cfg', 'dnn_model\yolov4-tiny.weights')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load objects from classes.txt
classes = []

with open('dnn_model\classes.txt', 'r') as file:
    for class_name in file:
        class_name = class_name.strip()
        classes.append(class_name)

# Focal length finder function
def focal_length_finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width

    return focal_length

# Distance finder function
def Distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame

    return distance

# Create face_data function to detect face
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_data(img):
    img_width = 0
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_detected = haar_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=6)

    for (x, y, w, h) in img_detected:
        img_rect = cv.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        cv.putText(img_rect, 'Face Detected', (x, y - 10), FONT, 1, GREEN, 2)
        img_width = w

    return img_width

# Initialize Image & calculate the focal length
person_img = cv.imread('Images\Person.jpg')
person_img_width = face_data(person_img)

person_img_estimated = focal_length_finder(KNOWN_DISTANCE, KNOWN_WIDTH, person_img_width)

# cv.putText(person_img, f'Distance: {person_img_estimated}', (50, 50), FONT, 1, BLUE, 2)
# cv.imshow('Face Detected', person_img)
print('Focal Length = ', person_img_estimated)

# Initialize Video
capture = cv.VideoCapture(0)

capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    isTrue, frame = capture.read()
    frame_distance_width = face_data(frame)
    class_ids, scores, bboxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    for (class_id, score, bbox) in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        frame_rect = cv.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)
        cv.putText(frame, class_name, (x, y - 10), FONT, 1, BLUE, 2)

        if frame_distance_width != 0:
            frame_distance_estimated = Distance_finder(person_img_estimated, KNOWN_WIDTH, frame_distance_width)

            cv.putText(frame_rect, f"'s Distance: {frame_distance_estimated}", (x + 120, y - 10), FONT, 1, BLUE, 2)
        
            if frame_distance_estimated <= 30:
                cv.putText(frame, f'Warning! You are too close to the camera. Please stay back!', (100, 360), FONT, 1, RED, 2)

    cv.imshow('Detect and Estimate', frame)

    if cv.waitKey(30) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()