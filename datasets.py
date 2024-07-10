from torch.utils.data import Dataset, DataLoader
import os 
import glob 
import cv2 

class Cityscapes(Dataset):
    def __init__(self, data_dir:str, train:bool, transforms:None)->None:
        self.data_dir = data_dir 
        self.train = train 
        self.transforms = transforms 

        split = 'train' if self.train else 'val'
        img_folder_path = os.path.join(data_dir, 'leftImage8bit', split)
        label_folder_path = os.path.join(data_dir, 'gtFine', split)

        self.img_fns = sorted(glob.glob(os.path.join(img_folder_path, '*/*.png'))) # includes full path
        self.labels = sorted(glob.glob(os.path.join(label_folder_path, '*/*labelIds.png')))

    def __len__(self):
        return len(self.img_fns)
    
    def __getitem__(self, idx):
        img_path = self.img_fns[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mask = cv2.imread(self.labels[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return (image, mask)
    