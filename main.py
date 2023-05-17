from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt


def main():
    model = get_model("fasterrcnn", 4, 1)
    logs = train(model, "/Users/aayushmaanjain/Documents/UCSD/spring23/cse252d/project/NBA-Player-Detector", 5)
    loss_metrics = np.array(list(logs.values()))
    print(loss_metrics.shape)
    training_loss = loss_metrics.sum(axis=0)
    plt.plot(training_loss)
    plt.show()
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()