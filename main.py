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
    NUM_EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    MODEL_NAME = config['model_name']
    MODEL_SAVE_AS = config['save_model_as']
    #-------------------------------#

    print("Loading dataset...")
    trainset, valset, testset = load_data(DATASET_PATH)

    print("Building model...")
    model = get_model(MODEL_NAME, trainable_backbone_layers=TRAINABLE_LAYERS)

    print("Training model...")
    modelpath = MODEL_SAVE_AS + ".pth"
    logs = train(model, trainset, valset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    plt.plot(logs['train_loss'], label='train loss')
    plt.plot(logs['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()