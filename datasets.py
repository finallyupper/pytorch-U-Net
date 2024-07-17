from torch.utils.data import Dataset, DataLoader
import os 
import glob 
import cv2 
import torch as th 
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

class CityscapesDataset(Dataset):
    def __init__(self, data_dir:str, train:bool, resize:tuple=(256, 512), split:str='train')->None:
        self.data_dir = data_dir 
        self.train = train 
        self.resize = resize 
        self.split = split
        
        img_folder_path = os.path.join(data_dir, 'leftImg8bit', split) 
        label_folder_path = os.path.join(data_dir, 'gtFine', split) 

        self.img_fns = sorted(glob.glob(os.path.join(img_folder_path, '*/*.png'))) # includes full path
        self.labels = sorted(glob.glob(os.path.join(label_folder_path, '*/*_labelIds.png')))

        background_idx = 0
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 33, -1]
        self.valid_classes = [background_idx, 7, 23, 26]
        self.background_idx = background_idx

        class_names = ['unlabelled', 'road', 'sky', 'car']

        colors = [   
                    [ 255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [255, 255, 0]
                ]

        self.class_mappings = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.num_classes = len(self.valid_classes)
        self.class_mapping_rgbs = dict(zip(range(self.num_classes), colors))

    def __len__(self):
        return len(self.img_fns)
    

    def enc_mask(self, mask):
        """Assigns mask IDs to background index and valid class indices"""
        mask_img = th.zeros((mask.size()[-2], mask.size()[-1]), dtype=th.uint8) # height, width
        for void_class in self.void_classes: # assign to background index 
            mask_img[mask == void_class] = self.background_idx
        for valid_class in self.valid_classes:
            mask_img[mask == valid_class] = self.class_mappings[valid_class] 

        return mask_img 
    
    def dec_mask(self, mask):
        """Decodes the mask to be RGB image"""
        rgb_img = th.zeros((3, mask.size()[-2], mask.size()[-1]), dtype=th.uint8)  

        for i in range(0, self.num_classes):
            rgb_img[0][mask == i] = self.class_mapping_rgbs[i][0]  # R
            rgb_img[1][mask == i] = self.class_mapping_rgbs[i][1]  # G
            rgb_img[2][mask == i] = self.class_mapping_rgbs[i][2]  # B

        return rgb_img

    def transforms(self, image, mask, height, width, filenames=None):
        """Transforms the image and mask and returns results of encoding and decodings"""
        # Resize image and Convert to tensor
        resized_image = image.resize((width, height), Image.BILINEAR)
        image = F.to_tensor(resized_image) 

        resized_mask = mask.resize((width, height), Image.NEAREST) 
        mask = np.asarray(resized_mask, dtype=np.int64)  
        mask = th.from_numpy(mask)

        mask_class = self.enc_mask(mask)
        mask_rgb = self.dec_mask(mask_class)

        if filenames is not None:
            return image, mask_class.long(), mask_rgb.long(), filenames
        else:
            return image, mask_class.long(), mask_rgb.long()
        
    def __getitem__(self, idx):
        img_path = self.img_fns[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(self.labels[idx]).convert('L') 
        height = self.resize[0]
        width = self.resize[1]

        if self.split == 'test':
            return self.transforms(image, mask, height, width, img_path)
        else:
            return self.transforms(image, mask, height, width)



            


