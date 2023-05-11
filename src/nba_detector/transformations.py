# The purpose of this module is to provide an interface to select different types of transformations
# Examples: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transformation(format:str = "pascal_voc") -> A.Compose:
    """
    Retrieves an example of a transformation, applies ToTensor in the end to 
    provide a tensor to the model.
    - format (str): any of ["pascal_voc", "albumentations", "coco", "yolo"]
    """
    if format not in ["pascal_voc", "albumentations", "coco", "yolo"]:
        raise ValueError(f"Format not recognized, it was sent {format}")
    
    transformation = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format=format, label_fields=['bounding_box_labels']))

    return transformation

"""
def show(sample):
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    import torch
    import PIL.Image

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)

    color_per_class = {}
    color_per_class[1] = "red"
    color_per_class[2] = "blue"
    color_per_class[3] = "yellow"
    color_per_box = [color_per_class[id.item()] for id in target["labels"]]
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors=color_per_box, width=4)
    
    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()



if __name__ == "__main__":

    #model = get_model()
    from dataset import BasketballDataset
    import torchvision

    PATH_TO_DATASET = './NBA-Player-Detector-1/'
    torchvision.disable_beta_transforms_warning()

    transformation = A.Compose([
        A.HorizontalFlip(p=1),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bounding_box_labels']))

    train_dataset = BasketballDataset(root_dir=PATH_TO_DATASET, transform=transformation, image_set='train')
    print(f"len of dataset is {len(train_dataset)}")
    image, targets = train_dataset[0]
    show((image, targets)) # run in debug to show the plot
    x = 0
"""