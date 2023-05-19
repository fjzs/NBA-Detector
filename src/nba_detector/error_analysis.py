from evaluate import evaluate_batch
from visualization import visualize_one_image
import torch
import cv2
import numpy as np

def analyse_one_image(model:torch.nn.Module, image:torch.Tensor, ground_truth:dict) -> np.ndarray:
    """
    image: torch.tensor of shape (C x H x W) and dtype uint8.

    prediction: dict(Tensor) containing:
        boxes: (Tensor) – Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax)
            format. Note that the boxes are absolute coordinates with respect to the image. 
            In other words: 0 <= xmin < xmax < W and 0 <= ymin < ymax < H.
        labels: (List[str]) – List containing the labels of bounding boxes.
    
    ground_truth: dict(Tensor) containing:
        boxes: (Tensor) – Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax)
            format. Note that the boxes are absolute coordinates with respect to the image. 
            In other words: 0 <= xmin < xmax < W and 0 <= ymin < ymax < H.
        labels: (List) – List containing the labels of bounding boxes.
    """


    mAP_dict = evaluate_batch(model, image.unsqueeze(0), [ground_truth])
    prediction = model(image.unsqueeze(0))[0]
    vis_image = visualize_one_image(image.byte(), prediction, ground_truth)

    return vis_image, mAP_dict

def save_analysis(vis_image: np.ndarray, mAP_dict: dict, save_path:str):
    """
    vis_image: np.ndarray of shape (H x W x C) and dtype uint8.
    mAP_dict: dict containing the mAP scores for each class.
    save_path: str
    """
    cv2.imwrite(save_path, vis_image)
    with open(save_path.replace('.jpg', '.txt'), 'w') as f:
        f.write(str(mAP_dict))
    return

if __name__ == "__main__":
    from create_model import get_model_fasterrcnn
    
    model = get_model_fasterrcnn(num_classes=3)

    image = torch.rand(3, 512, 512)

    gt_boxes = torch.tensor([[20, 50, 200, 200], [210, 210, 360, 480], [220, 220, 300, 310]], dtype=torch.float)
    gt_labels = torch.tensor([0,1,2])
    ground_truth = {}
    ground_truth['boxes'] = gt_boxes
    ground_truth['labels'] = gt_labels

    vis_image, mAP_dict = analyse_one_image(model, image, ground_truth)

    save_analysis(vis_image, mAP_dict, 'test.jpg')

    print('Done')