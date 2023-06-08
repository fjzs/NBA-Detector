# The purpose of this module is to provide an interface to select different types of transformations
# Examples: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformation(transformations: dict, format: str = "pascal_voc") -> A.Compose:
    """
    Retrieves an example of a transformation, applies ToTensor in the end to 
    provide a tensor to the model.

    - transformations(dict): transformation with its probability
    - format (str): any of ["pascal_voc", "albumentations", "coco", "yolo"]
    """
    if format not in ["pascal_voc", "albumentations", "coco", "yolo"]:
        raise ValueError(f"Format not recognized, it was sent {format}")

    assert type(transformations) == dict
    for key, p in transformations.items():
        assert 0 <= p <= 1

    # https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/
    transformation_list = []
    if 'horizontal_flip' in transformations:
        prob = transformations['horizontal_flip']
        transformation_list.append(A.HorizontalFlip(p=prob))
    if 'brightness_contrast' in transformations:
        prob = transformations['brightness_contrast']
        transformation_list.append(A.RandomBrightnessContrast(
            p=prob, brightness_limit=0.3, contrast_limit=0.3))
    if 'hue_saturation_value' in transformations:
        prob = transformations['hue_saturation_value']
        transformation_list.append(A.HueSaturationValue(p=prob))
    if 'defocus' in transformations:
        prob = transformations['defocus']
        transformation_list.append(A.Defocus(p=prob))
    if 'solarize' in transformations:
        prob = transformations['solarize']
        transformation_list.append(A.Solarize(p=prob))
    if 'sharpen' in transformations:
        prob = transformations['sharpen']
        transformation_list.append(A.Sharpen(p=prob))
    if 'equalize' in transformations:
        prob = transformations['equalize']
        transformation_list.append(A.Equalize(p=prob))
    if 'rgbshift' in transformations:
        prob = transformations['rgbshift']
        transformation_list.append(A.RGBShift(p=prob))

    ONE_OF_PROBABILITY = transformations['one_of_probability']
    assert 0 <= ONE_OF_PROBABILITY <= 1

    transformation = A.Compose(
        [
            A.OneOf(transformation_list, p=ONE_OF_PROBABILITY),
            ToTensorV2()  # this always goes in the end
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=[
                                 'bounding_box_labels'])
    )

    return transformation


def show(image, target, image_aug, target_aug):

    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
    image = F.convert_dtype(image, torch.uint8)

    if isinstance(image_aug, PIL.Image.Image):
        image_aug = F.to_image_tensor(image_aug)
    image_aug = F.convert_dtype(image_aug, torch.uint8)

    # Draw bounding boxes
    color_per_class = {}
    color_per_class[1] = "red"
    color_per_class[2] = "blue"
    color_per_class[3] = "yellow"
    color_per_box = [color_per_class[id.item()] for id in target["labels"]]
    color_per_box_aug = [color_per_class[id.item()]
                         for id in target_aug["labels"]]
    annotated_image = draw_bounding_boxes(
        image, target["boxes"], colors=color_per_box, width=3).permute(1, 2, 0).numpy()
    annotated_image_aug = draw_bounding_boxes(
        image_aug, target_aug["boxes"], colors=color_per_box_aug, width=3).permute(1, 2, 0).numpy()

    # Plot the images
    plt.subplot(1, 2, 1)
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(annotated_image_aug)
    plt.axis('off')
    plt.title('Augmented image')

    # Show the plot
    plt.subplots_adjust(left=0.05, right=0.95,
                        bottom=0.05, top=0.95, wspace=0.05)
    plt.margins(0, 0)
    plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    import torch
    import PIL.Image
    from dataset import BasketballDataset
    import torchvision
    import yaml

    PATH_TO_DATASET = './NBA-Player-Detector-1/'
    torchvision.disable_beta_transforms_warning()

    # ------------------- Config -------------------#
    config_file = './config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())['transformations']
        print(f"\nConfig file is:\n{config}\n")
    # -----------------------------------------------#

    # transformation = A.Compose([
    #    A.HorizontalFlip(p=1),
    #    ToTensorV2()
    # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bounding_box_labels']))
    transformation = get_transformation(config)

    train_dataset = BasketballDataset(
        root_dir=PATH_TO_DATASET, transform=None, image_set='train')
    train_dataset_aug = BasketballDataset(
        root_dir=PATH_TO_DATASET, transform=transformation, image_set='train')

    N = len(train_dataset)
    for i in range(N):
        image, targets = train_dataset[i]
        image_aug, targets_aug = train_dataset_aug[i]
        # run in debug to show the plot
        show(image, targets, image_aug, targets_aug)
