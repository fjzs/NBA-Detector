import os
import xml.etree.ElementTree as ET
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import numpy as np
import torch
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from typing import Tuple, List



def download_dataset_from_roboflow(format: str = 'voc', version_id: int = 1) -> None:
    """Download the dataset from Roboflow website using API call

        Parameters
        ----------
        format : str
            Format of the dataset to be downloaded

        Returns
        -------
        dataset : Dataset
        """
    assert type(version_id) == int, f"version_id is not int, it is {version_id}"
    assert 1 <= version_id <= 3, f"version_id has to be >=1 and <=3, it is {version_id}"

    from roboflow import Roboflow
    rf = Roboflow(api_key='NASBxoDeYCFInyN1wXD2')
    project = rf.workspace("francisco-zenteno-uryfd").project("nba-player-detector")
    project.version(version_id).download(format)


class BasketballDataset(Dataset):
    # File extensions
    XML_EXTENSION = '.xml'
    JPG_EXTENSION = '.jpg'
    # Keys defined for labels to be provided to the custom model
    BOXES_KEY: str = 'boxes'
    LABELS_KEY: str = 'labels'
    FILEPATH_KEY: str = 'filepath'
    # Labels to map detection objects to numbers
    LABEL_MAP: defaultdict(int) = {'ball': 1, 'player': 2, 'rim': 3}

    def __init__(self, root_dir: str, transform: A.Compose=None, image_set: str = 'train') -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_set = image_set
        self.image_ids: List[str] = self._get_image_ids()

    def __len__(self) -> int:
        """Returns count of dataset items

        Returns
        -------
        length : int
            Count of dataset items
        """
        return self.image_ids.__len__()

    def _get_image_ids(self) -> List[str]:
        """Iterate over the image set in the root folder to identify and list image and data files for the dataset

        Returns
        -------
        image_ids : list[str]
            List of ids made using the XML filename after removing extension
        """
        folder_path = os.path.join(self.root_dir, self.image_set)
        image_ids: List[str] = []
        for xml_file in os.listdir(folder_path):
            full_xml_path = os.path.join(folder_path, xml_file)
            if full_xml_path.endswith(BasketballDataset.XML_EXTENSION) and \
                    os.path.isfile(full_xml_path.replace(BasketballDataset.XML_EXTENSION, BasketballDataset.JPG_EXTENSION)):
                image_ids.append(os.path.join(
                    full_xml_path.removesuffix(BasketballDataset.XML_EXTENSION)))
        return image_ids

    def __getitem__(self, index) -> Tuple[Tensor, defaultdict[Tensor]]:
        """Returns the item at given index in the dataset

        Parameters
        ----------
        index : int
            Index of the dataset to be returned

        Returns
        -------
        image : Tensor
            Image at given index in the dataset as a Tensor
        target : defaultdict[Tensor]
            Dictionary containing keys boxes, labels, filepath. Value corresponding to boxes is of the form torch.Tensor[torch.Tensor],
            each containing bounding box coordinates in the form of [x0, y0, x1, y1], labels is of the form torch.Tensor, each value
            corresponding to an integer defined by LABEL_MAP constant, and filename is of the form str, specifying filepath of the original image.
        """

        img_path = self.image_ids[index] + BasketballDataset.JPG_EXTENSION
        ann_path = self.image_ids[index] + BasketballDataset.XML_EXTENSION
        pil_image = Image.open(img_path).convert('RGB') # This is PIL Image
        targets = self._get_annotations(ann_path)

        if self.transform:
            # Albumentations expects np.ndarray of shape (H,W,C)
            image_np = np.array(pil_image) # This is a numpy array of shape (H, W, C)
            # the parameter 'bounding_box_labels' in self.transform has to have the same name as when defined in
            # the compose function. For example:
            #transformation = A.Compose([
            #        A.HorizontalFlip(p=1),
            #        ToTensorV2()
            #    ],
            #    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bounding_box_labels']))


            # There is a bug in the bounding boxes, some of them come with an x_max > WIDTH or
            # y_max > HEIGHT, so we are going to trim those bounding boxes here for simplicity
            number_of_boxes = len(targets['labels'])
            if number_of_boxes > 0:
                HEIGHT, WIDTH, _ = image_np.shape
                height_vector = torch.ones(number_of_boxes) * HEIGHT
                width_vector = torch.ones(number_of_boxes) * WIDTH
                # Boxes come in this format: [x_min, y_min, x_max, y_max]

                # Replace x_max if it is > WIDTH
                targets["boxes"][:,2] = torch.min(targets["boxes"][:,2], width_vector)

                # Replace y_max if it is > HEIGHT
                targets["boxes"][:,3] = torch.min(targets["boxes"][:,3], height_vector)

            transformed = self.transform(image=  image_np,
                                         bboxes = targets["boxes"],
                                         bounding_box_labels = targets['labels'])
            image = transformed['image']
            # Transform the boxes to Tensors, because they are retrieved as list of tuples
            targets["boxes"] = Tensor(transformed['bboxes'])

        else:
            # If there was no transform, we need to transform the image to tensor (C,H,W)
            image = pil_to_tensor(pil_image) # This is a Tensor now of shape (C,H,W)

        return image, targets

    def _get_annotations(self, xml_file_path: str) -> defaultdict[Tensor]:
        """Define format of annotations for the dataset from a given file

        Parameters
        ----------
        xml_file_path : str
            Path to the XML file corresponding to a valid image and id

        Returns
        -------
        targets : defaultdict[Tensor]
            Dictionary containing keys boxes, labels, filepath. Value corresponding to boxes is of the form torch.Tensor[torch.Tensor],
            each containing bounding box coordinates in the form of [x0, y0, x1, y1], labels is of the form torch.Tensor, each value
            corresponding to an integer defined by LABEL_MAP constant, and filename is of the form str, specifying filepath of the original image.
        """
        # Initialize targets
        targets = defaultdict(list)
        # Parse XML file
        tree = ET.parse(xml_file_path)
        # Obtain root of the file
        root = tree.getroot()
        # Iterate over each object in the file and access required keys
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            label = obj.find('name').text.lower().strip()
            # Store data in required format in targets
            targets[BasketballDataset.BOXES_KEY].append(
                [xmin, ymin, xmax, ymax])
            targets[BasketballDataset.LABELS_KEY].append(
                BasketballDataset.LABEL_MAP[label])

        # Convert lists to Tensors
        targets[BasketballDataset.BOXES_KEY] = Tensor(
            targets[BasketballDataset.BOXES_KEY])
        targets[BasketballDataset.LABELS_KEY] = LongTensor(
            targets[BasketballDataset.LABELS_KEY])
        # Store file path
        targets[BasketballDataset.FILEPATH_KEY] = xml_file_path.removesuffix(
            BasketballDataset.XML_EXTENSION)
        return targets


def load_data(folder_name: str, train_transform: A.Compose = None, dataset_type: str = 'voc'):
    """Load the dataset from specified folder

    Parameters
    ----------
    folder_name : str
        Path to root directory containing the dataset
    train_transform: A.Compose object
        Comes from Albumentations
    dataset_type: 'voc'

    Returns
    -------
    tuple[BasketballDataset]
        List of Dataset object containing train, validation and test dataloader objects
    """
    os.listdir(folder_name)
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

    if dataset_type == 'coco':
        # Get annotation file paths
        train_ann_file = TRAIN_FOLDER + '/_annotations.coco.json'
        val_ann_file = VALID_FOLDER + '/_annotations.coco.json'
        test_ann_file = TEST_FOLDER + '/_annotations.coco.json'

        assert os.path.isfile(
            train_ann_file), "No annotations file in train folder"
        assert os.path.isfile(
            val_ann_file), "No annotations file in validation folder"
        assert os.path.isfile(
            test_ann_file), "No annotations file in test folder"

    # Load data
    train_dataset = BasketballDataset(root_dir=folder_name, transform=train_transform, image_set='train')
    val_dataset = BasketballDataset(root_dir=folder_name, image_set='valid')
    test_dataset = BasketballDataset(root_dir=folder_name, image_set='test')

    return train_dataset, val_dataset, test_dataset
