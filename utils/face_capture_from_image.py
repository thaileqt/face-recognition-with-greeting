import torch
from facenet_pytorch.models import mtcnn
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
# load image from file path with RGB mode
img = Image.open('IMG_9039.jpg').convert('RGB')

# resize image to below 1000
if img.size[0] > 1000:
    img = img.resize((1000, int(img.size[1] * 1000 / img.size[0])))
if img.size[1] > 1000:
    img = img.resize((int(img.size[0] * 1000 / img.size[1]), 1000))

# convert image to frame for mtcnn to detect
frame = np.array(img)
# detect faces
boxes, _ = mtcnn.detect(frame)

if boxes is not None:
    for box in boxes:
        x, y, w, h = box.astype(int)
        face = frame[y:h, x:w, :]
        filename = f"../data/Hoang/hoang1.jpg"
        try:
            cv2.imwrite(filename, face)
        except:
            pass