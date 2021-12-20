# import torch
# import torch.nn as nn
# from collections import OrderedDict

# # device 
# cpu = torch.device('cpu')
# gpu = torch.device('cuda:0')

# class C1(nn.Module):
#     def __init__(self):
#         super(C1, self).__init__()

#         # 3 channels are here: RGB
#         self.c1 = nn.Sequential(OrderedDict([
#             ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
#             ('relu1', nn.ReLU()),
#             ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
#         ]))

#     def forward(self, img):
#         output = self.c1(img)
#         return output


# class C2(nn.Module):
#     def __init__(self):
#         super(C2, self).__init__()

#         self.c2 = nn.Sequential(OrderedDict([
#             ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
#             ('relu2', nn.ReLU()),
#             ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
#         ]))

#     def forward(self, img):
#         output = self.c2(img)
#         return output


# # class C3(nn.Module):
# #     def __init__(self):
# #         super(C3, self).__init__()

# #         self.c3 = nn.Sequential(OrderedDict([
# #             ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
# #             ('relu3', nn.ReLU())
# #         ]))

# #     def forward(self, img):
# #         output = self.c3(img)
# #         return output


# class F4(nn.Module):
#     def __init__(self):
#         super(F4, self).__init__()

#         self.f4 = nn.Sequential(OrderedDict([
#             ('f4', nn.Linear(120, 84)),
#             ('relu4', nn.ReLU())
#         ]))

#     def forward(self, img):
#         output = self.f4(img)
#         return output


# class F5(nn.Module):
#     def __init__(self):
#         super(F5, self).__init__()

#         self.f5 = nn.Sequential(OrderedDict([
#             ('f5', nn.Linear(84, 10)),
#             ('sig5', nn.LogSoftmax(dim=-1))
#         ]))

#     def forward(self, img):
#         output = self.f5(img)
#         return output

        
        
# class LeNet5(nn.Module):
#     """
#     Input - 3x32x32
#     Output - 10
#     """
#     def __init__(self):
#         super(LeNet5, self).__init__()

#         self.c1 = C1()
#         self.c2 = C2()
#         #self.c2_1 = C2() 
#         #self.c2_2 = C2() 
#         #self.c3 = C3() 
#         self.f4 = F4() 
#         self.f5 = F5() 

#     def forward(self, img):
#         output = self.c1(img)

#         #x = self.c2_1(output)
#         #output = self.c2_2(output)

#         #output += x

#         #output = self.c3(output)
#         #output = output.view(img.size(0), -1)
#         output = self.c2(output)
#         output = self.f4(output)
#         output = self.f5(output)
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()