from datetime import datetime

import cv2
import os
from facenet_pytorch import MTCNN
import torch

def face_capture(name: str, image_count: int = 10):
    # Set up the device
    device = torch.device('cpu')

    # Initialize the MTCNN model
    mtcnn = MTCNN(keep_all=True, device=device)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Set the camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create a directory to store the captured faces
    name = name.strip().lower().replace(' ', '')
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.exists('data/'+name):
        os.mkdir('data/'+name)


    # Define a counter for the number of captured faces
    counter = 0

    while counter<image_count:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Detect faces using the MTCNN model

        try:
            boxes, _ = mtcnn.detect(frame)
        except:
            continue

        # Draw bounding boxes around the detected faces and save them to a data folder
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box.astype(int)
                face = frame[y:h, x:w, :]
                # save as datetime for image
                dt = datetime.timestamp(datetime.now())
                filename = f"data/{name}/{dt}.jpg"
                try:
                    cv2.imwrite(filename, face)
                except:
                    continue
                counter += 1
                cv2.rectangle(frame, (x, y), (w,h), (0, 255, 0), 2)

        # Display the resulting frame
        # cv2.imshow('frame', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the display window
    cap.release()
    cv2.destroyAllWindows()
