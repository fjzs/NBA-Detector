import torch
from visualization import CLASS_COLORS, drawrect
import cv2
import pandas as pd
import os


def apply(model, dataloader, folder_to_save):

    # Data to gather to fill in a df
    img_ids = []
    tps = []
    fps = []
    fns = []

    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
        print(f"Folder created: {folder_to_save}")

    id = 1
    for (images, labels) in dataloader:
        predictions = model(images)
        N = len(images)
        for i in range(N):
            image = images[i]
            gt = labels[i]
            pred = predictions[i]
            tp, tn, fp, fn, vis_image = analyze_single_image(image, gt, pred)
            id_name = str(id).zfill(4)
            img_ids.append(id_name)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)

            # Save the image
            image_path = os.path.join(folder_to_save, id_name + '.jpg')
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, vis_image)
            id += 1
    
    # Assemble the df
    data = {"img_id": img_ids, 
            "tp": tps, 
            "fp": fps, 
            "fn": fns}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(folder_to_save, 'df.csv'))
    print(f"df saved!")



def analyze_single_image(image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, MIN_PRED_SCORE=0.5):
    """

    Args:
        image (torch.Tensor): torch.float32 shape (C x H x W)
        gt (torch.Tensor): dictionary containing:
            - 'boxes'  -> Tensor of shape (N, 4)
            - 'labels' -> Tensor of shape (N, ) with ints
        pred (torch.Tensor): dictionary containing:
            - 'boxes'  -> Tensor of shape (M, 4)
            - 'labels' -> Tensor of shape (M, ) with ints
            - 'scores' -> Tensor of shape (M, ) with floats
    """

    assert image.dtype == torch.float32
    assert type(gt) == dict
    assert type(pred) == dict

    # Transform image to numpy array of ints and shape (H,W,C)
    vis_image = (image * 255).to(torch.uint8)
    vis_image = vis_image.permute(1,2,0).numpy()

    # Define the list of gt and pred to process
    # TODO check if to apply NMS here
    
    # Predicted boxes
    pred_indices = [i for i in range(len(pred['boxes'])) if pred['scores'][i] >= MIN_PRED_SCORE]
    pred_boxes = [pred['boxes'][i] for i in pred_indices]
    pred_labels = [pred['labels'][i] for i in pred_indices]
    
    # Ground truth
    gt_boxes = gt['boxes'].detach().clone().tolist()
    gt_labels = gt['labels'].detach().clone().tolist()

    # Calculate TP, TN, FP, FN
    tp, tn, fp, fn = calculate_tp_tn_fp_fn(gt_boxes, pred_boxes)

    # Draw predicted boxes
    for i in pred_indices:
        x_min, y_min, x_max, y_max = pred_boxes[i]
        point1 = (int(x_min), int(y_min))
        point2 = (int(x_max), int(y_max))
        label_id = pred_labels[i].item()
        color = CLASS_COLORS[label_id]
        drawrect(vis_image, point1, point2, color, thickness=4, style='dotted')
    
    # Draw ground truth boxes
    for i in range(len(gt_boxes)):
        x_min, y_min, x_max, y_max = gt_boxes[i]
        point1 = (int(x_min), int(y_min))
        point2 = (int(x_max), int(y_max))
        label_id = gt_labels[i]
        color = CLASS_COLORS[label_id]
        vis_image = cv2.rectangle(vis_image, point1, point2, color, thickness=4)
    
    return tp, tn, fp, fn, vis_image





def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Calculates IOU between 2 boxes

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        iou (float):
    """
    assert len(box1) == 4
    assert len(box2) == 4

    # Calculate the intersection coordinates
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate the intersection area
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    #print(f"intersection_area: {intersection_area}")

    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    #print(f"union_area: {union_area}")

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

    # Unit test: [x_min, y_min, x_max, y_max]
    """
    box1 = [0,0,5,6]
    box2 = [1,3,7,8]
    print(f"IOU should be 0.25: {calculate_iou(box1, box2)}")
    box1 = [0,0,5,6]
    box2 = [5,3,7,8]
    print(f"IOU should be 0: {calculate_iou(box1, box2)}")
    box1 = [0,0,5,6]
    box2 = [-10,3,0,8]
    print(f"IOU should be 0: {calculate_iou(box1, box2)}")
    del box1, box2
    """

def calculate_tp_tn_fp_fn(gt_boxes: list, pred_boxes: list, iou_threshold=0.5):
    """Calculates TP, TN, FP, FN

    Args:
        gt_boxes (list): N elements, each element is a list of 4 numbers
        pred_boxes (list): M elements, each element is a list of 4 numbers
        iou_threshold (float, optional): Defaults to 0.5.

    Returns:
        TP, TN, FP, FN: ints
    """
    assert type(gt_boxes) == list
    assert type(pred_boxes) == list

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Create a copy of the original data to lists
    gt_box_list = gt_boxes.copy()
    pr_box_list = pred_boxes.copy()

    # Iterate through each predicted box
    for pred_box in pr_box_list:
        max_iou = 0
        max_iou_index = -1

        # Find the ground truth box with the highest IoU
        for i, gt_box in enumerate(gt_box_list):
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i

        # Check if the highest IoU is above the threshold
        if max_iou >= iou_threshold:
            tp += 1
            # Remove the matched ground truth box
            gt_box_list.pop(max_iou_index)
        else:
            fp += 1

    # Any remaining ground truth boxes are false negatives
    fn = len(gt_box_list)

    # Calculate true negatives
    tn = 0

    return tp, tn, fp, fn


"""
if __name__ == "__main__":
    
    # Load Model
    from create_model import get_model
    from dataset import load_data
    model = get_model("fasterrcnn", num_classes=4, trainable_backbone_layers=1)
    model_path = "C://Users//zente//Downloads//model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)

    # Load dataset
    from train_model import collate_fn
    dataset_path = "./NBA-Player-Detector-1"
    trainset, valset, testset = load_data(dataset_path)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=1, 
        drop_last=True,
        collate_fn=collate_fn)

    apply(model, trainloader, "./EA/")
"""