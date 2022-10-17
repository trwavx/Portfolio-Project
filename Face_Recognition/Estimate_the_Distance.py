import cv2 as cv

KNOWN_DISTANCE = 40
KNOWN_WIDTH = 15

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0 ,255)

FONT = cv.FONT_HERSHEY_COMPLEX

faces_haarCascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def faces_data(img):
    img_width = 0
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_dtc = faces_haarCascade.detectMultiScale(img_gray, scaleFactor=1.6, minNeighbors=6)
    
    for (x, y, w, h) in faces_dtc:
        cv.rectangle(img, (x, y), (x + w, y + h), BLUE, 2)
        
        img_width = w
        
    return img_width

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    
    return focal_length

def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    
    return distance

img = cv.imread(r'Images/Vu Truong Anh/1.jpg')

img_width = faces_data(img)
img_focal_length = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, img_width)

print(f'Focal Length = {img_focal_length}')

capture = cv.VideoCapture(0)

capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    isTrue, frame = capture.read()
    frame_width = faces_data(frame)
    
    if frame_width != 0:
        frame_distance_finder = distance_finder(img_focal_length, KNOWN_WIDTH, frame_width)
        
        cv.putText(frame, f'Distance = {round(frame_distance_finder, 2)}', (50, 50), FONT, 1, RED, 2)
        
        if frame_distance_finder <= 30:
            cv.putText(frame, 'YOU ARE TOO CLOSE! PLEASE STAND BACK!', (300, frame.shape[0] // 2), FONT, 1, RED, 2)
            
    cv.imshow('Estimate the Distance w/ OpenCV', frame)
    
    if cv.waitKey(30) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()