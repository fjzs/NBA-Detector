import albumentations as A
from src.nba_detector.transformations import get_transformation
from src.nba_detector.dataloader import load_data
import pytest

def test_get_dataset():
    trainloader, valloader, testloader = load_data('/content/NBA-Player-Detector-1/', 224, 224, 4)

    assert(trainloader != None)
    assert(testloader != None)
    assert(valloader != None)