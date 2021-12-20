from models import ResNet, resnet18, FCN32, ResNet18, FCN16
from torch.utils.data import Dataset
from torch import nn
import os
import torchvision

import torch
import torch.optim as optim

from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score

import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


from data_preprocessing import FCNDATA
from os import listdir
from os.path import isfile, join

from distutils.version import LooseVersion
import torch.nn.functional as F

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n = 1
    c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss



batch_size = 1
num_epochs = 50

# split the dataset into train/val/test in 70-15-15 alphabetically.
img_path = './data_semantics/training/image_2/'
gt_path = './data_semantics/training/semantic/'



onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
onlyfiles = sorted(onlyfiles)
print(onlyfiles)

train_files = onlyfiles[:140]
print(len(train_files))
val_files = onlyfiles[140:170]
print(len(val_files))
test_files = onlyfiles[170:]

print(len(test_files))

trainset = FCNDATA(train_files)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

valset = FCNDATA(train_files)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = FCNDATA(test_files)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# model = FCN32(backbone='resnet18')

# model_download = torchvision.models.resnet18(pretrained=True)
# model_instance = ResNet18(model_download) 

print("Model Summary ")
# print(model_instance.parameters)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# total_params = get_n_params(model_instance)
# # print("Total params: ", total_params)


# 3. Define a Loss function and optimizer
import torch.optim as optim










# 4. Train the network
def train(path, model_instance):
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001)#, momentum=0.9)
    optimizer = optim.SGD(model_instance.parameters(), lr=0.1)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print("input shape : ", inputs.shape)
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model_instance(inputs)
            print("outputs.shape : ", outputs.shape, labels.shape)
            outputs = torch.argmax(outputs, dim=1)
            #loss = criterion(outputs.long(), labels.long())
            loss = cross_entropy2d(outputs, labels)
            loss.backward()
            optimizer.step() # let's comment this and see

    # Let’s quickly save our trained model:
    torch.save(model_instance.state_dict(), path)
    
def test(PATH, model_instance):
    # now load the model and test it.
    net = model_instance()
    net.load_state_dict(torch.load(PATH))


    ### Let us look at how the network performs on the whole dataset.
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    overall_predicted = []
    overall_grounds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            overall_grounds = overall_grounds + list(labels.cpu().detach().numpy())
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            overall_predicted = overall_predicted + list(predicted.cpu().detach().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    confusion_matrix_test = cm(overall_grounds, overall_predicted)
    print("*"*60)
    print('Accuracy of the network on the 5000 test images: %d %%' % (100 * correct / total))
    print('F1 score on test dataset: ', f1_score(overall_grounds, overall_predicted, average='weighted'))
    print("Confusion Matrix on Test dataset : ", confusion_matrix_test)




if __name__ == "__main__":
    model_download = torchvision.models.resnet18(pretrained=True)
    model_instance = ResNet18(model_download) # this is fcn32 with resnet18 as backbone
    print(model_instance.parameters)
    PATH = './fcn32_net.pth'
    train(PATH, model_instance)
    test(PATH, model_instance)
    
    
    # try with the FCN16
    model_instance = FCN16(model_download)
    PATH = './fcn16_net.pth'
    train(PATH, model_instance)
    test(PATH, model_instance)