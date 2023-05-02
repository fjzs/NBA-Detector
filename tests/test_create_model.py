from src.nba_detector.create_model import get_model
import torch
import pytest


""" def test_get_model_good_input():
    model_names = ["fasterrcnn"]
    num_classes = [2]
    trainable_layers = [1]
    for m in model_names:
        for n in num_classes:
            for t in trainable_layers:
                model = get_model(m,n,t)
                assert (type(model) == torch.nn.Module)
 """

def test_get_model_bad_format():
    bad_model_name = "Fasterrcnn"
    with pytest.raises(ValueError):
        get_model(bad_model_name, 2)
 