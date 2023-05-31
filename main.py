from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
from src.nba_detector.dataset import load_data
import matplotlib.pyplot as plt
import yaml


def main():
    #--------- Config -------------#
    config_file = './config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())['train']
        print(f"\nConfig file is:\n{config}\n")

    DATASET_PATH = config['dataset_path']
    TRAINABLE_LAYERS = config['trainable_layers']
    MODEL_NAME = config['model_name']
    #-------------------------------#

    print("Loading dataset...")
    trainset, valset, testset = load_data(DATASET_PATH)

    print("Building model...")
    model = get_model(MODEL_NAME, trainable_backbone_layers=TRAINABLE_LAYERS)

    print("Training model...")
    logs = train(model, trainset, valset, config)

    plt.plot(logs['train_loss'], label='train loss')
    plt.plot(logs['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()