from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
from src.nba_detector.dataset import load_data
import pandas as pd
import torch.utils.data
import yaml


def main():
    #--------- Config -------------#
    config_file = './config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())
        print(f"\nConfig file is:\n{config['train']}\n")

    DATASET_PATH = config['train']['dataset_path']
    TRAINABLE_LAYERS = config['train']['trainable_layers']
    NUM_EPOCHS = config['train']['epochs']
    BATCH_SIZE = config['train']['batch_size']
    MODEL_NAME = config['train']['model_name']
    #-------------------------------#

    print("Loading dataset...")
    trainset, valset, testset = load_data(DATASET_PATH)

    print("Building model...")
    model = get_model("fasterrcnn", trainable_backbone_layers=TRAINABLE_LAYERS)

    print("Training model...")
    modelpath = MODEL_NAME + ".pth"
    logs = train(model, modelpath, trainset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # Save the logs to a df
    df = pd.DataFrame.from_dict(logs)
    df.to_csv("logs.csv")
    print(f"Logs saved to logs.csv")    


if __name__ == '__main__':
    main()