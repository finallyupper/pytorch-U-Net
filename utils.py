import yaml
import wandb

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_log(configs):
    wandb.login()
    wandb.init(project=configs['project'],
               entity=configs['entity'],
               name =configs['name']
            #    config={
            #         "batch_size": BATCH_SIZE,
            #         "img_size": IMG_SIZE,
            #         "epochs": EPOCHS,
            #         "num_classes": NUM_CLASS,
            #         "learning_rate": LR,
            #         "model_name": MODEL_NAME}
                    )