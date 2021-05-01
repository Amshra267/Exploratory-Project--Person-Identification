# PyTorch
from torchsummary import summary
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.datasets as dset
from torch import optim
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import albumentations as A
from albumentations.pytorch import ToTensorV2

from collections import defaultdict

import random
from pathlib import Path
import sys
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from siamese_facenet import SiameseNetwork

"""### DATA LOADING AND PREPROCESSING"""

### here i will create the list of paths related to each one

train_list = []
test_list = []

root = '../input/cleaned-lfw/cleaned_lfw'
for i in os.listdir(root):
    imgs = glob(os.path.join(root, i, "*"))
    length = len(imgs)
    if (length==7):
        ratio = 0.7
    else:
        ratio=0.8
    train_div = int(ratio*length)
    train_list.append(imgs[0:train_div])
    test_list.append(imgs[train_div:])

def lenght(list_of_lists):
    count = 0
    for i in range(len(list_of_lists)):
        count +=len(list_of_lists[i])
    return count

class trainingDataset(Dataset):#Get two images and whether they are related.
    
    def __init__(self, data_list, transform=None):
        self.list = data_list
        self.transform = transform
        
    def get_pair(self, data_list, positive):
        pair = []
        if positive:
            value = random.randint(0,255)
            id = [value, value]
            label = 1
        else:
            label = 0
            while True:
                id = [random.randint(0,255), random.randint(0,255)]
                if id[0] != id[1]:
                    break
        
        for i in range(2):
            filepath = ''
            while True:
                sub = self.list[id[i]]
                filepath = random.choice(sub)
                if not os.path.exists(filepath):
                    continue
                break
            pair.append(filepath)
        return pair, torch.tensor(label).long()
  
    def __getitem__(self,index):
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        path, label = self.get_pair(self.list, should_get_same_class)
        img0 = cv2.imread(path[0])
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.imread(path[1])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)


        if self.transform is not None:
            img0 = self.transform(image=img0)["image"]
            img1 = self.transform(image=img1)["image"]
        
        img0 = img0/255.
        img1 = img1/255.
        return img0, img1, label 
    
    def __len__(self):
        return lenght(self.list)#essential for choose the num of data in one epoch

epochs = 600
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT+2, width=IMAGE_WIDTH+4, p=1),
    A.RandomCrop(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,p=1),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10),
    A.OneOf([A.HueSaturationValue(p=0.4), A.RGBShift(p=0.4)], p=0.4),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
])
val_transform = A.Compose([
                            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, p=1),
                            ToTensorV2()
                        ])



trainset = trainingDataset(train_list , transform=train_transform)
train_loader = DataLoader(trainset,
                        shuffle=True,#whether randomly shuffle data in each epoch, but cannot let data in one batch in order.
                        num_workers=8,
                        batch_size=128)

valset = trainingDataset(test_list, transform=val_transform)
validation_loader = DataLoader(valset,
                        shuffle=True,#whether randomly shuffle data in each epoch, but cannot let data in one batch in order.
                        num_workers=8,
                        batch_size=128)

"""## VISUALISATION

For visulaisation batch size= 8 else 128
"""

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    
    
dataiter = iter(train_loader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())



summary(SiameseNetwork().cuda(), [(3,32,32), (3,32,32)])

model = SiameseNetwork()
device = torch.device('cuda:0')
model = model.to(device)

criterion = nn.CrossEntropyLoss() # computes softmax and then the cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0005)

lambda1 = lambda iteration: ((0.0001 * iteration + 1) ** -0.75)

exp_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


iteration = 0

if os.path.exists('logs'):
    checkpoint = torch.load('logs')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print('Restore model at epoch - ', epoch)

history = defaultdict(list) ## for storing histories
history['val_loss'].append(np.inf)
for i in range(epochs):
    val_acc = []
    val_loss = []
    train_acc = []
    train_loss = []
    
    
    model.train() # setting for training
    for batch_idx, data in enumerate(train_loader):
        
        
        img0, img1 , labels = data #img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)#move to GPU
        optimizer.zero_grad()
        
        logits = model(img0, img1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        _, preds = torch.max(logits, 1)
        total = len(labels)
        correct = torch.sum(preds==labels)
        train_acc.append(correct.item()/total)
        iteration +=1
        print('Iteration: {}, learning rate: {:.5f}, Loss: {:.4f}, Accuracy:{:.3f}'.format(iteration, 
                                                                                       optimizer.param_groups[0]["lr"], 
                                                                                       loss.item(), correct.item()/total))
        exp_lr_scheduler.step()
    model.eval() # setting for training
    
    for batch_idx, data in enumerate(validation_loader):
        
        img0, img1 , labels = data #img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)#move to GPU
        
        logits = model(img0, img1)
        loss = F.cross_entropy(logits, labels, reduction = 'sum')/len(img0)

        val_loss.append(loss.item())
        _, preds = torch.max(logits, 1)
        total = len(labels)
        correct = torch.sum(preds==labels)
        val_acc.append(correct.item()/total)  
    
    epoch_t_loss = np.mean(train_loss)
    epoch_t_acc = np.mean(train_acc)
    epoch_v_loss = np.mean(val_loss)
    epoch_v_acc = np.mean(val_acc)
    print('Epoch: {}, Loss: {:.4f}, Accuracy:{:.3f}, Val Loss: {:.4f}, Val Accuracy: {:.3f}'.format(
        i+1,  epoch_t_loss, epoch_t_acc , epoch_v_loss,  epoch_v_acc))
    
    history['loss'].append(epoch_t_loss)
    history['val_loss'].append(epoch_v_loss)
    history['acc'].append(epoch_t_acc)
    history['val_acc'].append(epoch_v_acc)
    if (history['val_loss'][-1]<min(history['val_loss'][:-1])):
        print('val_loss_decreased from {:.4f} to {:.4f}, saving_checkpoint for epoch {}'.format(min(history['val_loss'][:-1]), 
                                                                                                history['val_loss'][-1], i+1))
        torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': exp_lr_scheduler.state_dict(),
                }, 'logs')

plt.figure(figsize=(5,5))
ax1 = plt.subplot(2,1,1)
ax1.plot(history['loss'], label='train_loss')
ax1.plot(history['val_loss'][1:], label='val_loss')
ax1.legend()
ax2 = plt.subplot(2,1,2)
ax2.plot(history['acc'], label='train_acc')
ax2.plot(history['val_acc'][1:], label='val_acc')
ax2.legend()
plt.savefig('metrics.png')

if os.path.exists('logs'):
    checkpoint = torch.load('logs')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print('Restore model at epoch - ', epoch)

test_acc = []
test_loss = []
for batch_idx, data in enumerate(validation_loader):

    img0, img1 , labels = data #img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
    img0, img1 , labels = img0.to(device), img1.to(device) , labels.to(device)#move to GPU

    logits = model(img0, img1)
    loss = F.cross_entropy(logits, labels, reduction = 'sum')/len(img0)

    test_loss.append(loss.item())
    _, preds = torch.max(logits, 1)
    total = len(labels)
    correct = torch.sum(preds==labels)
    test_acc.append(correct.item()/total)  

t_loss = np.mean(test_loss)
t_acc = np.mean(test_acc)
print('Test Loss: {:.4f}, Test Accuracy: {:.3f}'.format(t_loss,  t_acc))

import matplotlib.lines as mlines

def cmc2(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = np.array(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
#     print(m,n)
    indices = np.argsort(-distmat, axis=1)
#     print(indices)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            print(gids)
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & 1)#_unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    print('I am here')
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries

default_color = ['r','g','b','c','m','y','orange','brown']
default_marker = ['*','o','s','v','X','*','.','P']

class CMC:
    def __init__(self,cmc_dict, color=default_color, marker = default_marker, name = None):
        self.color = color
        self.marker = marker
        self.cmc_dict = cmc_dict
        self.name = name
        
    def plot(self,title,rank=10, xlabel='Rank',ylabel='Matching Rates',show_grid=True):        
        fig, ax = plt.subplots()
        fig.suptitle(title)
        x = list(range(1, rank+1, 1))
        plt.ylim(0, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank+1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc)+1))
                
            if name == list(self.cmc_dict.keys())[-1]:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[0], marker=self.marker[0], label= name)
            else:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i+1], marker=self.marker[i+1], label=name)
                i = i+1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)
        plt.savefig(self.name)
        #plt.show()
    
    def save(self, title, filename, 
             rank=20, xlabel='Rank',
             ylabel='Matching Rates (%)', show_grid=True,
             save_path=os.getcwd(), format='png', **kwargs):
        fig, ax = plt.subplots()
        fig.suptitle(title)
        x = list(range(0, rank+1, 5))
        plt.ylim(0, 1.0)
        plt.xlim(1, rank)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x)
        plt.grid(show_grid)

        method_name = []
        i = 0
        for name in self.cmc_dict.keys():
            if rank < len(self.cmc_dict[name]):
                temp_cmc = self.cmc_dict[name][:rank]
                r = list(range(1, rank+1))
            else:
                temp_cmc = self.cmc_dict[name]
                r = list(range(1, len(temp_cmc)+1))
                
            if name == list(self.cmc_dict.keys())[-1]:
                globals()[name] = mlines.Line2D(r, temp_cmc, color='r', marker='*', label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
            else:
                globals()[name] = mlines.Line2D(r, temp_cmc, color=self.color[i], marker=self.marker[i], label='{:.1f}% {}'.format(self.cmc_dict[name][0]*100, name))
                i = i+1
            ax.add_line(globals()[name])
            method_name.append(globals()[name])

        plt.legend(handles=method_name)
        plt.show()

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

    return out[0][1].item()

    
test_idx = np.arange(0,256)
print('len of iestidx - ', len(test_idx))


score_matrix=np.zeros((256,256))
for i in range(len(test_idx)):
    img1=random.choice(test_list[i])
    for j in range(len(test_idx)):
        img2=random.choice(test_list[j])
        score_matrix[i][j]=score(img1,img2)
        #print(score_matrix[i][j])
print('completed computing score matrix')
#calculating cmc score
cmc_score=cmc2(score_matrix,gallery_ids=test_idx,query_ids=test_idx, topk = 256)
print(cmc_score)
np.save('cmc_score1', cmc_score)
cmc_dict ={

'simaese_face_with_32X32': cmc_score,
}
curve=CMC(cmc_dict, name = 'Siamese_face_32X32_torch_cmc')
curve.plot(title = "", show_grid=False)

