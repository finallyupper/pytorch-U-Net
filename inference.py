from torch.utils.data import DataLoader
import torch as th
import os 
import glob 
from datasets import CityscapesDataset 
from model import UNET 
from utils import load_yaml
from PIL import Image 
import torch.nn.functional as F
import numpy as np 
from torchvision.transforms import functional as TF
import warnings
warnings.filterwarnings("ignore")

def save_prediction(mask, gt, save_path):
    colors = np.array([
        [255, 0, 0],    # Class 0
        [0, 255, 0],    # Class 1
        [0, 0, 255],    # Class 2
        [255, 255, 0]   # Class 3
    ], dtype=np.uint8)
    
    color_mask = colors[mask]  # Map each class index to its corresponding color
    color_mask = Image.fromarray(color_mask)

    gt = TF.to_pil_image(gt)
    concat_img = Image.new('RGB', (gt.width * 2, gt.height))
    concat_img.paste(gt, (0, 0))
    concat_img.paste(color_mask, (gt.width, 0))
    concat_img.save(save_path)

#TODO : Use threshold
def inference(model, device, test_loader, threshold=0.5, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, (images, targets, _, fns) in enumerate(test_loader):
        images, targets = images.to(device), targets.to(device)
        
        with th.no_grad():
            preds = model.forward(images).cpu() # shape: 1x4x256x512
            preds = F.interpolate(preds, size=(1024, 2048), mode='bilinear')
            images = F.interpolate(images, size=(1024, 2048), mode='bilinear')

            pred_masks = th.argmax(preds, dim=1).numpy()

            for i, (pred_mask, image) in enumerate(zip(pred_masks, images)):
                if i % 5 == 0:
                    save_path = os.path.join(save_dir, f'pred_{fns[i].split("/")[-1]}')
                    save_prediction(pred_mask, image, save_path)

def main():
    device = th.device('cuda:2' if th.cuda.is_available() else 'cpu')

    # Load configurations
    model_config = load_yaml('configs.yaml')
    inference_config = model_config['inference']

    # Load Dataset
    image_path = inference_config['data_root']
    image_fns = sorted(glob.glob(os.path.join(image_path, '*/*.png')))

    test_dataset = CityscapesDataset(data_dir = inference_config['data_root'], 
                                   train=False, 
                                   resize=tuple(inference_config['image_size']),
                                   split='test') 
    test_loader = DataLoader(test_dataset, 
                             batch_size=inference_config['batch_size'], 
                             shuffle=False)
    print('[INFO] Successfully loaded Dataset!') 

    # Load model
    unet = UNET(in_channels=3, num_classes=test_dataset.num_classes).to(device)
    state_dict = th.load(inference_config['model_path'])
    unet.load_state_dict(state_dict['model_state_dict'])
    unet = unet.to(device)
    print('[INFO] Successfully loaded model!')    

    inference(model=unet, device=device, test_loader=test_loader, threshold=inference_config['threshold'], save_dir=inference_config['save_path'])

if __name__ == '__main__':
    main()

