from model import *
from data_preprocessing import *

from torch.utils.data import Dataset

import os

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




num_epochs = 100

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
 
batch_size = 128

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset = STL10('./splits/train.txt')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# use 300 * 10 = 3,000 for validation and 500 * 10 = 5,000 images for testing]
valset = STL10('./splits/val.txt')
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = STL10('./splits/test.txt')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)



# load the pre-trained model
#print("Loading the pre-trained model !!!")
net = Net()


print("Model Summary ")
print(summary(net, (3, 32, 32)))

PRETRAINED_PATH = './stl10_net.pth'
#net.load_state_dict(torch.load(PRETRAINED_PATH))
#print("Pre-trained params ...", dict(net.named_parameters()))

print("#"*60)
print(net.parameters())
print("#"*60)

# for parameter in net.parameters():
#     print(parameter)
# print("#"*60)

# # Total params:  62006
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

total_params = get_n_params(net)
print("Total params: ", total_params)
# print("#"*60)

# 3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001)#, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 4. Train the network
def train():
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # let's comment this and see

            # print statistics
            running_loss += loss.item()
            if i % 9 == 8:    # 5,000/128 = 39 ; print every 9 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 9))
                running_loss = 0.0
        
        # Showing the loss
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        # check for accuracy after every 5 epochs on validation set.
        if epoch % 5 == 0:
            print("Testing the accuracy after epoch {} on validation set for accuracy ...".format(epoch))
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            overall_grounds = []
            overall_predicted = []
            
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    overall_grounds = overall_grounds + list(labels.cpu().detach().numpy())
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    
                    # calculate loss on validation set.
                    val_loss = criterion(outputs, labels)
                    writer.add_scalar('Loss/Val', val_loss.item(), epoch)
                    
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    overall_predicted = overall_predicted + list(predicted.cpu().detach().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print("*"*60)
            print('Accuracy of the network on the 3000 validation images: %d %%' % (100 * correct / total))
            #print("overall_grounds : ", overall_grounds)
            #print("overall_predicted: ", overall_predicted)
            confusion_matrix_val = cm(overall_grounds, overall_predicted)
            print('F1 score on val dataset: ', f1_score(overall_grounds, overall_predicted, average='weighted'))
            print("Confusion Matrix on Validation dataset : ", confusion_matrix_val)
            
            PATH_checkpoint = './checkpoints/stl10_net_' + str(epoch) + "_" + str((100 * correct / total))
            torch.save(net.state_dict(), PATH_checkpoint)
            
            
            writer.add_scalar('Accuracy/val', (100 * correct / total), epoch)
        
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        if epoch % 20 == 0:print()

    print('Finished Training')
    print("*"*60)

    # Let’s quickly save our trained model:
    PATH = './stl10_net.pth'
    torch.save(net.state_dict(), PATH)

# # 5. Test the network on the test data
# dataiter = iter(testloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

def test(PATH):
    # now load the model and test it.
    net = Net()
    net.load_state_dict(torch.load(PATH))

    #Okay, now let us see what the neural network thinks these examples above are:
    # outputs = net(images)

    # _, predicted = torch.max(outputs, 1)

    # print("*"*60)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                               for j in range(batch_size)))


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
    train()
    PATH = './stl10_net.pth'
    test(PATH)
    