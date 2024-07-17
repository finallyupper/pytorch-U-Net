from torch.utils.data import Dataset, DataLoader
import torch as th
from torch.nn import CrossEntropyLoss 
from torch.optim import Adam 
import os 
import glob 
import cv2 
from datasets import CityscapesDataset 
from model import UNET 
from utils import load_yaml
import time
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter

NUM_CLASSES = 4
device = th.device('cuda:2' if th.cuda.is_available() else 'cpu')

#TODO: Change the .yaml loading part
def train(device, model, train_loader, loss_function, optimizer):
    losses = []
    # switch to train mode
    model.train()

    totLoss = 0
    print('[INFO] training UNet ...')
    progress_bar = tqdm(train_loader, total=len(train_loader))

    for batch, (images, targets, _) in enumerate(progress_bar):
        images, targets = images.to(device), targets.to(device) 

        optimizer.zero_grad()

        preds = model.forward(images) # [4, 4, 256, 256]
        loss = loss_function(preds, targets) # targets : [4, 1, 256, 256]

        
        loss.backward()
        optimizer.step()

        totLoss += loss.item() 
        losses.append(loss.item())
        progress_bar.set_postfix({'loss': loss.item()})

    return totLoss, losses 

def valid(device, model, valid_loader, loss_function):
    # switch to evaluation mode
    model.eval()
    progress_bar = tqdm(valid_loader, total=len(valid_loader))
    totLoss = 0
    losses = []
    for i, (images, targets, _) in enumerate(progress_bar):
        images, targets = images.to(device), targets.to(device)

        with th.no_grad():
            preds = model.forward(images)

        loss = loss_function(preds, targets)
        totLoss += loss 
        losses.append(loss.item())
    return totLoss, losses


def main():
    model_config = load_yaml('configs.yaml')
    train_config = model_config['train'] 

    train_dataset = CityscapesDataset(data_dir = train_config['data_root'], train=True, resize=tuple(train_config['image_size']), split='train') 
    val_dataset = CityscapesDataset(data_dir = train_config['data_root'], train=False, resize=tuple(train_config['image_size']), split='val') 


    print(f'[INFO] {len(train_dataset)} samples in the training set...')
    print(f'[INFO] {len(val_dataset)} samples in the validation set...')

    train_Loader = DataLoader(train_dataset,
                             batch_size=train_config['batch_size'],
                             shuffle=True)
    valid_loader = DataLoader(val_dataset,
                             batch_size=train_config['batch_size'],
                             shuffle=False)    
    print('Successfully loaded datasets!')

    unet = UNET(in_channels=3, num_classes=NUM_CLASSES).to(device) # Check

    lossFunc = CrossEntropyLoss()
    opt = Adam(unet.parameters(), lr=train_config['learning_rate'])

    # calculate steps per epoch for training and test set
    trainSteps = len(train_dataset) // train_config['batch_size']
    testSteps = len(val_dataset) // train_config['batch_size']

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}
    best_test_avg_loss = 1000.0
    writer = SummaryWriter()

    for epoch in range(train_config['epochs']):
        print(f'[INFO] EPOCH {epoch + 1}/{train_config["epochs"]}')
        start = time.time()

        train_tot_loss, train_losses = train(device, unet, train_Loader, lossFunc, opt)
        test_tot_loss, test_losses = valid(device, unet, valid_loader, lossFunc)
        train_avg_loss = train_tot_loss / trainSteps 
        test_avg_loss = test_tot_loss / testSteps 

        H['train_loss'].append(train_avg_loss)
        H['test_loss'].append(test_avg_loss)

        end = time.time()
        print(f'Train loss: {train_avg_loss:.3f}, Test loss: {test_avg_loss:.3f}')

        print(f'[INFO] Took {((end - start) / 60):.3f} minutes for epoch {epoch}')

        time.sleep(2)
        writer.add_scalar("Loss/train", train_tot_loss, epoch) 
        writer.add_scalar("Loss/valid", train_tot_loss, epoch)  

        if test_avg_loss < best_test_avg_loss:
            best_test_avg_loss = test_avg_loss
            print(f'[INFO] Saving the best model...')
            model_name = f'unet_epoch_{epoch + 1}_loss_{test_avg_loss:.3f}.pth'
            th.save(unet.state_dict(), os.path.join(train_config["model_path"], model_name))
            print(f'[INFO] Model saved as {model_name}')
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()



