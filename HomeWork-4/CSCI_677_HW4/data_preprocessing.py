import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import PIL
from PIL import Image

# 1. Load and normalize STL-10 and not CIFAR10

# Training images: 5,000
# Test images: 800 *10 = 8,000 [use 300 * 10 = 3,000 for validation and 500 * 10 = 5,000 images for testing]

# Preprocessing
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(32),
     transforms.Normalize((0, 0, 0), (1, 1, 1))]) # for all 3 RGB channels


class STL10(Dataset):
    def __init__ (self, path):
        super().__init__()
        #path = './splits/train.txt'
        # Initialize some paths
        self.img_label_infos = path
        self.fd = open(self.img_label_infos, "r")
        
        self.images_labels_list = self.fd.read().splitlines()
        self.len = len(self.images_labels_list)
        
    
    def __len__ (self):
        # return the length of the full dataset .
        # You might want to compute this in
        # __init__ function and store it in a variable
        # and just return the variable , to avoid recomputation
        return self.len
    
    def get_img_label(self, image_label):
        return image_label[0], image_label[1]
    
    def __getitem__ (self, idx):
        # Somehow get the image and the label
        if idx >= self.len:
            print("Invalid Index for the image")
            return None
        
        image_label = self.images_labels_list[idx]
        image_label = image_label.split(" ")
        
        img_file , label = self.get_img_label(image_label)
        # read the image file
        img_pil = PIL.Image.open(img_file)
        img_transformed = transforms(img_pil)
        #print("label " , type(lint(label)), int(label))
        return img_transformed, torch.tensor(int(label))
    
# if __name__ == "__main__":
#     path = './splits/train.txt'
#     stl10 = STL10(path)
    