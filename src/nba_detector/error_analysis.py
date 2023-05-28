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
    
    # Load model
    from create_model import get_model
    from dataset import load_data
    model = get_model("fasterrcnn", num_classes=4, trainable_backbone_layers=1)
    model_path = "G:/My Drive/ACV Project/m_v3_50e.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load dataset
    from train_model import collate_fn
    dataset_path = "./NBA-Player-Detector-1"
    trainset, valset, testset = load_data(dataset_path)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=False, num_workers=1, drop_last=True,
        collate_fn=collate_fn)

    # Analyze the images
    for i, (images, labels) in enumerate(trainloader):
        print(f"Len of images: {len(images)}")
        print(f"Shape of images: {images[0].shape}")
        print(f"labels are:\n{labels}")
        preds = model(images)
        print(f"preds shape is: {preds.shape}")
        print(f"preds are:\n{preds}")
        
        image = images[0]
        label = labels[0]    

    gt_boxes = torch.tensor([[20, 50, 200, 200], [210, 210, 360, 480], [220, 220, 300, 310]], dtype=torch.float)
    gt_labels = torch.tensor([0,1,2])
    ground_truth = {}
    ground_truth['boxes'] = gt_boxes
    ground_truth['labels'] = gt_labels

    vis_image, mAP_dict = analyse_one_image(model, image, ground_truth)

    save_analysis(vis_image, mAP_dict, 'test.jpg')

    print('Done')
