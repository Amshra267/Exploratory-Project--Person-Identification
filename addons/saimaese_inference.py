# PyTorch
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from siamese_facenet import SiameseNetwork
import numpy as np
import cv2

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32


val_transform = A.Compose([
                            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, p=1),
                            ToTensorV2()
                        ])


model = SiameseNetwork()
device = torch.device('cuda:0')
model = model.to(device)
model.eval()

def score(img1,img2):

    image1 = cv2.imread(img1)
    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype(float)
    image1 = val_transform(image=image1)['image']/255
    image1 = torch.reshape(image1, (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)).float().cuda()
    image2 = cv2.imread(img2)
    image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB).astype(float)
    image2 = val_transform(image=image2)['image']/255.
    image2 = torch.reshape(image2, (1,3, IMAGE_HEIGHT, IMAGE_WIDTH)).float().cuda()

    out = model(image1, image2)

    return F.softmax(out)[0][1]



if __name__ == "__main__":
    if os.path.exists('logs/siamese_logs'):
        checkpoint = torch.load('logs/siamese_logs')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print('Restore model at epoch - ', epoch)
    print(score("sample_images/Tiger1.jpeg", "sample_images/Akki1.jpeg"))
