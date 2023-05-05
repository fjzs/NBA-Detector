import os
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets.coco import Optional
import torchvision.datasets as datasets
from pycocotools.coco import COCO

from roboflow import Roboflow

api_key = os.getenv('ROBOFLOW_KEY')
rf = Roboflow(api_key=api_key)
project = rf.workspace(
    "francisco-zenteno-uryfd").project("nba-player-detector")
dataset = project.version(1).download("coco")


class BasketballDataset(datasets.CocoDetection):
    BBOX_KEY: str = 'bbox'
    BOXES_KEY: str = 'boxes'
    CATEGORY_ID_KEY: str = 'category_id'
    LABELS_KEY: str = 'labels'
    IMAGES_KEY: str = 'images'
    FILEPATH_KEY: str = 'filepath'

    def __init__(self, root: str, annFile: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        ann = json.load(open(annFile, 'r'))
        self.file_names: List[str] = []
        if BasketballDataset.IMAGES_KEY in ann:
            for image in ann[BasketballDataset.IMAGES_KEY]:
                if 'file_name' in image:
                    self.file_names.append(image['file_name'])

    def __getitem__(self, index: int) -> Tuple[any, defaultdict[torch.Tensor], str]:
        img, label = super().__getitem__(index)

        modified_target = defaultdict(list)
        for ann in label:
            if BasketballDataset.BBOX_KEY in ann:
                modified_target[BasketballDataset.BOXES_KEY].append(
                    ann[BasketballDataset.BBOX_KEY])
            if BasketballDataset.CATEGORY_ID_KEY in ann:
                modified_target[BasketballDataset.LABELS_KEY].append(
                    ann[BasketballDataset.CATEGORY_ID_KEY])
        modified_target[BasketballDataset.BOXES_KEY] = torch.Tensor(
            modified_target[BasketballDataset.BOXES_KEY])
        modified_target[BasketballDataset.LABELS_KEY] = torch.LongTensor(
            modified_target[BasketballDataset.LABELS_KEY])
        modified_target[BasketballDataset.FILEPATH_KEY] = self.file_names[index]
        return img, modified_target


def load_data(folder_name: str, width: int, height: int, batch_size: int):
    """Load the dataset from specified folder

    Parameters
    ----------
    folder_name : str
        Path to root directory containing the dataset
    width : int
        Width of transformed image
    height : int
        Height of transformed image
    batch_size : int
        Batch size for training dataset

    Returns
    -------
    list[DataLoader]
        List of DataLoader object containing train, validation and test dataloader objects
    """
    assert os.path.isdir(folder_name), "Given path does not exist"
    assert set({'train', 'test', 'valid'}).issubset(
        set(os.listdir(folder_name))), "train, test, valid folders not present in path"

    # Create folder paths
    TRAIN_FOLDER: str = folder_name + '/train'
    VALID_FOLDER: str = folder_name + '/valid'
    TEST_FOLDER: str = folder_name + '/test'

    assert os.path.isdir(TRAIN_FOLDER), "No folder named train"
    assert os.path.isdir(VALID_FOLDER), "No folder named valid"
    assert os.path.isdir(TEST_FOLDER), "No folder named test"

    # Get annotation file paths
    train_ann_file = TRAIN_FOLDER + '/_annotations.coco.json'
    val_ann_file = VALID_FOLDER + '/_annotations.coco.json'
    test_ann_file = TEST_FOLDER + '/_annotations.coco.json'

    assert os.path.isfile(
        train_ann_file), "No annotations file in train folder"
    assert os.path.isfile(
        val_ann_file), "No annotations file in validation folder"
    assert os.path.isfile(test_ann_file), "No annotations file in test folder"

    assert type(width) is int, "width is not an integer"
    assert type(height) is int, "height is not an integer"
    assert type(batch_size) is int, "batch_size is not an integer"

    # Create transform to resize images into required shape
    transform = transforms.Compose([transforms.Resize((width, height)),
                                    transforms.ToTensor()])

    # Load data
    train_dataset = BasketballDataset(root=TRAIN_FOLDER,
                                      annFile=train_ann_file,
                                      transform=transform)
    val_dataset = BasketballDataset(root=VALID_FOLDER,
                                    annFile=val_ann_file,
                                    transform=transform)
    test_dataset = BasketballDataset(root=TEST_FOLDER,
                                     annFile=test_ann_file,
                                     transform=transform)

    # Create DataLoader objects
    # trainloader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True)
    # valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset


# Example of function call
# trainloader, valloader, testloader = load_data('/content/NBA-Player-Detector-1/', 224, 224, 4)
