# The purpose of this module is to provide an interface to select different types of transformations
# Examples: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

import albumentations as A
import cv2

def get_transformation(format:str = "coco") -> A.Compose:
    """
    Retrieves an example of a transformation
    - format (str): any of ["pascal_voc", "albumentations", "coco", "yolo"]
    """
    if format not in ["pascal_voc", "albumentations", "coco", "yolo"]:
        raise ValueError(f"Format not recognized, it was sent {format}")
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ], 
        bbox_params=A.BboxParams(format = format))
    return transform

if __name__ == "__main__":
    output = get_transformation()
    print(output)