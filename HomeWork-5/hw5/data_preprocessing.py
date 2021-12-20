import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import PIL
from PIL import Image
import cv2
from os.path import isfile, join

img_path = './data_semantics/training/image_2/'
gt_path = './data_semantics/training/semantic/'

# Preprocessing
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(32),
     transforms.Normalize((0, 0, 0), (1, 1, 1))]) # for all 3 RGB channels


class FCNDATA(Dataset):
    def __init__ (self, images_name_list):
        super().__init__()
        self.images_name_list = images_name_list
        self.len = len(self.images_name_list)       
    
    def __len__ (self):
        return self.len
    
    def __getitem__ (self, idx):
        # Somehow get the image and the label
        if idx >= self.len:
            print("Invalid Index for the image")
            return None
        
        img_name = join(img_path, self.images_name_list[idx])
        gt_name = join(gt_path, self.images_name_list[idx])
        img = cv2.imread(img_name)
        #img = torch.tensor(img)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img).float()
        
        gt_img = cv2.imread(gt_name)
        #gt_img = torch.tensor(gt_img)
        gt_img = gt_img.transpose(2,0,1)
        gt_img = torch.from_numpy(gt_img).float()
        
        
        return img, gt_img
        
    
