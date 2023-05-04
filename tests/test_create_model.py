from src.nba_detector.create_model import get_model
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
import pytest


def test_get_model_fasterrcnn():
    num_classes = [2]
    trainable_layers = [1]
    for n in num_classes:
        for t in trainable_layers:
            model = get_model("fasterrcnn", n, t)
            assert (type(model) == FasterRCNN)

def test_get_model_bad_name():
    bad_model_name = "Fasterrcnn"
    with pytest.raises(ValueError):
        get_model(bad_model_name, 2)

def test_get_model_bad_num_classes():
    good_model_name = "fasterrcnn"
    bad_num_classes = [None, -1, 0, 1]
    for x in bad_num_classes:
        with pytest.raises(ValueError):
            get_model(good_model_name, x)

def test_get_model_bad_trainable_layers():
    good_model_name = "fasterrcnn"
    good_num_classes = 2
    bad_trainable_num = [-1,6]
    for x in bad_trainable_num:
        with pytest.raises(ValueError):
            get_model(good_model_name, good_num_classes, x)
