import json

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
from playsound import playsound
from unidecode import unidecode


def face_match(frame, model_path):  # img_path= location of photo, data_path= location of data.pt

    # getting embedding matrix of the given img
    face, prob = mtcnn(frame, return_prob=True)  # returns cropped face and probability

    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

    saved_data = torch.load(model_path)  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))


mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # initializing resnet for face img to embeding conversion
# Open the default camera
cap = cv2.VideoCapture(0)

threshold = 0.7
current_people = []
disappear_frame_count = {}
# Loop through the video frames
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(frame)
    appear_this_frame = []

    # Draw bounding boxes around the faces and save to file
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2, :]
            if face is None:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            try:
                result = face_match(face, 'data.pt')
                if result[1] < threshold:
                    text_config = json.load(open('text.json', 'r'))
                    name = result[0]
                    cv2.putText(frame, f'{name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    appear_this_frame.append(result[0])
                    if result[0] not in current_people:
                        playsound('speechs/' + result[0] + '.wav')
                        current_people.append(result[0])
                        disappear_frame_count[result[0]] = 0
            except Exception as e:
                print(e)
    for people in current_people:
        if people not in appear_this_frame:
            disappear_frame_count[people] += 1
            if disappear_frame_count[people] > 5:
                current_people.remove(people)
                disappear_frame_count.pop(people)

    # Display the output frame
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
