from src.nba_detector.create_model import get_model
import torch
import pytest


def test_get_model_good_input():
    model_names = ["fasterrcnn"]
    num_classes = [2]
    trainable_layers = [1]
    for m in model_names:
        for n in num_classes:
            for t in trainable_layers:
                model = get_model(m,n,t)
                assert (type(model) == torch.nn.Module)


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