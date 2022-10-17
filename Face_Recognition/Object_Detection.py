import cv2 as cv

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

FONT = cv.FONT_HERSHEY_COMPLEX

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

net = cv.dnn.readNet('dnn_model/yolov4-tiny.cfg', 'dnn_model/yolov4-tiny.weights')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []

with open('dnn_model/classes.txt', 'r') as file:
    for class_name in file.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# capture = cv.VideoCapture('Videos/Traffic_1.mp4')
capture = cv.VideoCapture('Videos/Traffic_2.mp4')

while True:
    isTrue, frame = capture.read()
    class_ids, scores, bboxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    for (class_id, scores, bbox) in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        
        cv.putText(frame, class_name, (x, y - 10), FONT, 1, GREEN, 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        
    cv.imshow('Object Detection w/ OpenCV', frame)
    
    if cv.waitKey(30) & 0xFF==27:
        break

capture.release()
cv.destroyAllWindows()