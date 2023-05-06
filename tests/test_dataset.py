from collections import defaultdict
import os

from src.nba_detector.dataset import load_data, download_dataset_from_roboflow, BasketballDataset

from torch import Tensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

PATH_TO_DATASET = './NBA-Player-Detector-1/'


def test_download_dataset():
    download_dataset_from_roboflow()
    assert (os.path.isdir(PATH_TO_DATASET))
    assert (os.path.isdir(os.path.join(PATH_TO_DATASET, 'train')))
    assert (os.path.isdir(os.path.join(PATH_TO_DATASET, 'test')))
    assert (os.path.isdir(os.path.join(PATH_TO_DATASET, 'valid')))


def test_basketball_dataset_train():
    train_dataset = BasketballDataset(
        root_dir=PATH_TO_DATASET, image_set='train')

    assert train_dataset is not None


def test_basketball_dataset_valid():
    valid_dataset = BasketballDataset(
        root_dir=PATH_TO_DATASET, image_set='valid')

    assert valid_dataset is not None


def test_basketball_dataset_test():
    test_dataset = BasketballDataset(
        root_dir=PATH_TO_DATASET, image_set='test')

    assert test_dataset is not None


def test_get_dataset():
    train_dataset, val_dataset, test_dataset = load_data(
        PATH_TO_DATASET, transforms.Compose([transforms.ToTensor()]))

    assert train_dataset is not None
    assert test_dataset is not None
    assert val_dataset is not None


def test_make_dataloader():
    train_dataset, val_dataset, test_dataset = load_data(
        PATH_TO_DATASET, transforms.Compose([transforms.ToTensor()]))
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    assert trainloader is not None
    assert testloader is not None
    assert valloader is not None


def test_dataloader_datatypes():
    train_dataset, val_dataset, test_dataset = load_data(
        PATH_TO_DATASET, transforms.Compose([transforms.ToTensor()]))
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    for ds in [trainloader.dataset, valloader.dataset, testloader.dataset]:
        for data in ds:
            img, label = data
            assert (type(img) == Tensor)
            assert (type(label) == defaultdict)


def test_target_keys():
    train_dataset, val_dataset, test_dataset = load_data(
        PATH_TO_DATASET, transforms.Compose([transforms.ToTensor()]))
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    for ds in [trainloader.dataset, valloader.dataset, testloader.dataset]:
        for data in ds:
            _, label = data
            assert 'boxes' in label
            assert 'labels' in label
            assert 'filepath' in label
