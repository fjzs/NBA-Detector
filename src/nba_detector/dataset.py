import os
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Tuple, List

import torch
import torchvision.transforms as transforms
from torchvision.datasets.coco import Optional
import torchvision.datasets as datasets

from roboflow import Roboflow

def download_dataset_from_roboflow() -> None:
    rf = Roboflow(api_key='NASBxoDeYCFInyN1wXD2')
    project = rf.workspace(
        "francisco-zenteno-uryfd").project("nba-player-detector")
    dataset = project.version(1).download("coco")


class BasketballDataset(datasets.CocoDetection):
    # Keys from default Cocodetection module in labels variable
    IMAGES_KEY: str = 'images'
    BBOX_KEY: str = 'bbox'
    CATEGORY_ID_KEY: str = 'category_id'
    # Keys defined for labels to be provided to the custom model
    BOXES_KEY: str = 'boxes'
    LABELS_KEY: str = 'labels'
    FILEPATH_KEY: str = 'filepath'

    def __init__(self, root: str, annFile: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None) -> None:
        """Modified to store filepath of each image file. This will be required during testing and individual image evaluation to pinpoint issues.
        """
        super().__init__(root, annFile, transform, target_transform, transforms)
        # Read the annotations file as a json object
        ann: json = json.load(open(annFile, 'r'))
        self.file_names: List[str] = []
        # Append filepath of all images into filenames list
        if BasketballDataset.IMAGES_KEY in ann:
            for image in ann[BasketballDataset.IMAGES_KEY]:
                if 'file_name' in image:
                    self.file_names.append(image['file_name'])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, defaultdict[torch.Tensor]]:
        """Returns the item at given index in the dataset

        Parameters
        ----------
        index : int
            Index of the dataset to be returned

        Returns
        -------
        image : torch.Tensor
            Image at given index in the dataset as a Tensor
        modified_target : defaultdict[torch.Tensor]
            Dictionary containing keys boxes, labels, filepath. Value corresponding to boxes is of the form torch.Tensor[torch.Tensor], 
            each containing bounding box coordinates in the form of [x0, y0, x1, y1], labels is of the form torch.Tensor, each value
            corresponding to category_id in the dataset, and filename is of the form str, specifying filepath of the original image.
        """
        image, label = super().__getitem__(index)

        modified_target = defaultdict(list)
        # Iterate over all labels
        for ann in label:
            if BasketballDataset.BBOX_KEY in ann:
                # Modify [x0, y0, w, h] format of coco dataset to [x0, y0, x1, y1] for PascalVOC
                ann[BasketballDataset.BBOX_KEY][2] += ann[BasketballDataset.BBOX_KEY][0]
                ann[BasketballDataset.BBOX_KEY][3] += ann[BasketballDataset.BBOX_KEY][1]
                # Append data to box key in target dictionary
                modified_target[BasketballDataset.BOXES_KEY].append(
                    ann[BasketballDataset.BBOX_KEY])
            if BasketballDataset.CATEGORY_ID_KEY in ann:
                # Append data to labels key in target dictionary
                modified_target[BasketballDataset.LABELS_KEY].append(
                    ann[BasketballDataset.CATEGORY_ID_KEY])

        # Convert lists to torch tensors
        modified_target[BasketballDataset.BOXES_KEY] = torch.Tensor(
            modified_target[BasketballDataset.BOXES_KEY])
        modified_target[BasketballDataset.LABELS_KEY] = torch.LongTensor(
            modified_target[BasketballDataset.LABELS_KEY])
        # Add filename to target dictionary
        modified_target[BasketballDataset.FILEPATH_KEY] = self.file_names[index]

        return image, modified_target


def load_data(folder_name: str, transform: transforms):
    """Load the dataset from specified folder

    Parameters
    ----------
    folder_name : str
        Path to root directory containing the dataset
    transform : transforms
        Transformations to apply to dataset

    Returns
    -------
    tuple[BasketballDataset]
        List of Dataset object containing train, validation and test dataloader objects
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

    return train_dataset, val_dataset, test_dataset


# Example of function call
# train_dataset, val_dataset, test_dataset = load_data(
#     '/NBA-Player-Detector-1/', transforms.Compose([transforms.ToTensor()]))
