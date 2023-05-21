from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
from src.nba_detector.dataset import load_data
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt


def main():
    model = get_model("fasterrcnn", 4, 1)
    path = "/Users/aayushmaanjain/Documents/UCSD/spring23/cse252d/project/NBA-Player-Detector"
    trainset, valset, testset = load_data(path)
    logs = train(model, trainset, valset, 5)
    loss_keys = [k for k in logs.keys() if k.startswith("loss_")]
    loss_metrics = np.array([logs[k] for k in loss_keys])
    print(loss_metrics.shape)
    training_loss = loss_metrics.sum(axis=0)
    print(f"Training loss: {training_loss}")
    
    # commented this lines apparently they dont work in colab
    #plt.plot(training_loss) 
    #plt.show()
    #plt.waitforbuttonpress()

if __name__ == '__main__':
    main()