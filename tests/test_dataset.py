from src.nba_detector.transformations import get_transformation
from src.nba_detector.dataset import load_data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def test_get_dataset():
    train_dataset, val_dataset, test_dataset = load_data(
        '/NBA-Player-Detector-1/', transforms.Compose([transforms.ToTensor()]))

    assert (train_dataset != None)
    assert (test_dataset != None)
    assert (val_dataset != None)


def test_make_dataloader():
    train_dataset, val_dataset, test_dataset = load_data(
        '/NBA-Player-Detector-1/', transforms.Compose([transforms.ToTensor()]))
    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    for data in trainloader.dataset:
        img, label = data
        print(label)
        assert (trainloader != None)
        assert (testloader != None)
        assert (valloader != None)
