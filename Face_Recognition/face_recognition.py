import numpy as np
from PIL import Image
import cv2 as cv
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

frame_size = (540, 960)
DATA_PATH = 'faces_dataset'
IMG_PATH = 'faces_dataset/faces_list'

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH + '/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH + '/faceslist.pth')
        
    names = np.load(DATA_PATH + '/usernames.npy')
    return embeds, names

def inference(model, face, local_embeds, threshold=3):
    embeds = []
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)
    
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)
    
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    print(min_dist * power, names[embed_idx])

    if min_dist * power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
        ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
        ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv.resize(img,(face_size, face_size), interpolation=cv.INTER_AREA)
    face = Image.fromarray(face)
    return face

if __name__ == '__main__':
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 540)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 960)
    
    embeddings, names = load_faceslist()
    
    while cap.isOpened():
        isSuccess, frame = cap.read()
        
        if isSuccess:
            boxes, props = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv.rectangle(frame, (bbox[0], bbox[1]), 
                                             (bbox[2], bbox[3]), (0, 0, 255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy() * power
                        frame = cv.putText(frame, names[idx] + '_{:.2f}'.format(score), 
                                            (bbox[0], bbox[1]), cv.FONT_HERSHEY_DUPLEX, 
                                            2, (0, 255, 0), 2, cv.LINE_8)
                    else:
                        frame = cv.rectangle(frame, (bbox[0],bbox[1]), 
                                              (bbox[2], bbox[3]), (0, 0, 255), 6)
                        frame = cv.putText(frame, 'Unknown', (bbox[0], bbox[1]), 
                                            cv.FONT_HERSHEY_DUPLEX, 2, 
                                            (0, 255, 0), 2, cv.LINE_8)


        cv.imshow('Face Recognition', frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()