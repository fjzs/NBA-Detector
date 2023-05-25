from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
from src.nba_detector.dataset import load_data
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
import yaml


def main():
    #--------- Config -------------#
    config_file = './config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())
        print(f"\nConfig file is:\n{config}\n")
    DATASET_PATH = config['dataset_path']
    TRAINABLE_LAYERS = config['trainable_layers']
    NUM_EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']
    #-------------------------------#

    print("Loading dataset...")
    trainset, valset, testset = load_data(DATASET_PATH)

    print("Building model...")
    model = get_model("fasterrcnn", trainable_backbone_layers=TRAINABLE_LAYERS)

    print("Training model...")
    logs = train(model, trainset, valset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    loss_keys = [k for k in logs.keys() if k.startswith("loss_")]
    loss_metrics = np.array([logs[k] for k in loss_keys])
    print(loss_metrics.shape)
    training_loss = loss_metrics.sum(axis=0)
    print(f"Training loss: {training_loss}")
    print(f"loss_metrics: {loss_metrics}")
    
    # commented this lines apparently they dont work in colab unless you are in a notebook
    #plt.plot(training_loss) 
    #plt.show()
    #plt.waitforbuttonpress()

if __name__ == '__main__':
    main()