
# def inference(model, device, test_loader, threshold=0.5, save_dir='predictions'):
#     os.makedirs(save_dir, exist_ok=True)
#     for batch_idx, (images, targets, _, fns) in enumerate(test_loader):
#         images, targets = images.to(device), targets.to(device)
        
#         with th.no_grad():
#             preds = model.forward(images).cpu() # shape: 1x4x256x512
#             preds = F.interpolate(preds, size=(1024, 2048), mode='bilinear')
#             images = F.interpolate(images, size=(1024, 2048), mode='bilinear')

#             targets = targets.unsqueeze(1).float()
#             targets = F.interpolate(targets, size=(1024, 2048), mode='nearest').cpu().numpy() 
#             targets = targets.squeeze(1).astype(np.uint8)

#             pred_masks = th.argmax(preds, dim=1).numpy()
        
#             acc_sum = (pred_masks == targets).sum()
#             acc = float(acc_sum) / (targets.shape[0]*targets.shape[1]*targets.shape[2])
#             print("Accuracy: ", acc)

#             for i, (pred_mask, image) in enumerate(zip(pred_masks, images)):
#                 save_path = os.path.join(save_dir, f'pred_{fns[i].split("/")[-1]}')
#                 save_prediction(pred_mask, image, save_path)

# def main():
#     device = th.device('cuda:2' if th.cuda.is_available() else 'cpu')

#     # Load configurations
#     model_config = load_yaml('/home/yoojinoh/Others/pytorch-U-Net/configs.yaml')
#     inference_config = model_config['inference']

#     # Load model
#     unet = UNET(in_channels=3, num_classes=NUM_CLASSES).to(device)
#     state_dict = th.load(inference_config['model_path'])
#     unet.load_state_dict(state_dict)
#     unet = unet.to(device)
    
#     print('[INFO] Successfully loaded model!')    
    
#     image_path = inference_config['data_root']
#     image_fns = sorted(glob.glob(os.path.join(image_path, '*/*.png')))

#     test_dataset = CityscapesDataset(data_dir = inference_config['data_root'], 
#                                    train=False, 
#                                    resize=tuple(inference_config['image_size']),
#                                    split='test') 
#     test_loader = DataLoader(test_dataset, 
#                              batch_size=inference_config['batch_size'], 
#                              shuffle=False)
    
#     inference(model=unet, device=device, test_loader=test_loader, threshold=inference_config['threshold'])

# if __name__ == '__main__':
#     main()