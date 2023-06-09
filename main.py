from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
from src.nba_detector.dataset import load_data
from src.nba_detector.transformations import get_transformation
import pandas as pd
import yaml


def main():
    #--------- Config -------------#
    config_file = './config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())
        print(f"\nConfig file is:\n{config}\n")

    # Train configurations
    DATASET_PATH = config['train']['dataset_path']
    TRAINABLE_LAYERS = config['train']['trainable_layers']
    NUM_EPOCHS = config['train']['epochs']
    BATCH_SIZE = config['train']['batch_size']    
    MODEL_NAME = config['train']['model_name']
    MODEL_SAVE_AS = config['train']['save_model_as']

    # transformations configurations
    transformation = get_transformation(config['transformations'])
    #-------------------------------#

    print("Loading dataset...")
    trainset, valset, testset = load_data(DATASET_PATH, train_transform=transformation)

    print("Building model...")
    model = get_model(MODEL_NAME, trainable_backbone_layers=TRAINABLE_LAYERS)

    print("Training model...")
    modelpath = MODEL_SAVE_AS + ".pth"
    logs = train(model, modelpath, trainset, valset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    # Save the logs to a df
    df = pd.DataFrame.from_dict(logs)
    filename = "log_" + MODEL_SAVE_AS + ".csv"
    df.to_csv(filename)
    print(f"Logs saved to {filename}")    


if __name__ == '__main__':
    main()