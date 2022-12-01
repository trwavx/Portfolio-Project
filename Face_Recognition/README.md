## USAGE
This is a FaceNet and MTCNN inference from [this Repository](https://github.com/timesler/facenet-pytorch).
1. Requirements:
    ```bash
    # Install with Pip
    !pip install numpy
    !pip install opencv-python
    !pip install torch torchvision torchaudio
    !pip install facenet-pytorch

    ```
1. Detection & Capturing:
    ```bash
    # Face Detection:
    python face_detect.py
    
    # Face Capturing (Remember to input your name FIRST in console):
    python face_capture.py

    ```
1. Create FaceList and Recognition:
    ```bash
    # Update FaceList:
    python update_faces.py
    
    # Face Recognition:
    python face_recognition.py

    ```