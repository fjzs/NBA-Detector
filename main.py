from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


def main():
    model = get_model("fasterrcnn", 4, 1)
    logs = train(model, "/Users/aayushmaanjain/Documents/UCSD/spring23/cse252d/project/NBA-Player-Detector", 1)

if __name__ == '__main__':
    main()