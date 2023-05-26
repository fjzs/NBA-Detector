import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# Reference: https://medium.com/data-science-at-microsoft/how-to-smoothly-integrate-meanaverageprecision-into-your-training-loop-using-torchmetrics-7d6f2ce0a2b3

def update_metric(metric:MeanAveragePrecision(), preds:list, targets:list) -> dict:
    """
    metric: An instance of MeanAveragePrecision()
    preds (List): 
    A list consisting of dictionaries each containing the key-values (each dictionary corresponds to a single image). 
    Parameters that should be provided per dict
    boxes: (FloatTensor) of shape (num_boxes, 4) containing num_boxes detection boxes of the format specified in the constructor. 
        By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
    scores: FloatTensor of shape (num_boxes) containing detection scores for the boxes.
    labels: IntTensor of shape (num_boxes) containing 0-indexed detection classes for the boxes.
    masks: bool of shape (num_boxes, image_height, image_width) containing boolean masks. Only required when iou_type=”segm”.

    targets (List): A list consisting of dictionaries each containing the key-values (each dictionary corresponds to a single image). 
    Parameters that should be provided per dict:
        boxes: FloatTensor of shape (num_boxes, 4) containing num_boxes ground truth boxes of the format specified in the constructor. 
            By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
        labels: IntTensor of shape (num_boxes) containing 0-indexed ground truth classes for the boxes.
    
    Returns:
        metric: An updated instance of MeanAveragePrecision()
    """
    assert(len(preds)==len(targets)), "Size mismatch between length of list of predictions and length of list of targets {} != {}".format(len(preds),len(targets))
    metric.update(preds,targets)
    return metric

def update_metric_on_batch(metric:MeanAveragePrecision(), model:torch.nn.Module, image_batch:torch.Tensor, targets:list) -> dict:
    """
    model: A PyTorch model of type torch.nn.Module returned from create_model.py
    image_batch: A batch of images of type torch.Tensor
    targets: A list of dictionaries corresponding to per-image targets.
        Each dictionary would have the following keys:
        boxes: FloatTensor of shape (num_boxes, 4) containing num_boxes ground truth boxes of the format specified in the constructor. 
            By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
        labels: IntTensor of shape (num_boxes) containing 0-indexed ground truth classes for the boxes.

    Returns:
        metric: An updated instance of MeanAveragePrecision()    
    """
    assert(len(image_batch)==len(targets)), "Size mismatch in Batch size and length of target list {} != {}".format(len(image_batch),len(targets))
    model.eval()
    with torch.no_grad():
        preds = model(image_batch)
        update_metric(metric,preds,targets)
    return metric

def update_metric_on_dataloader(metric:MeanAveragePrecision(), model:torch.nn.Module, dataloader:torch.utils.data.DataLoader) -> dict:
    """
    metric: An instance of MeanAveragePrecision()
    model: A PyTorch model of type torch.nn.Module returned from create_model.py
    dataloader: An instance of torch.data.utils.Dataloader

    Returns:
        metric: An updated instance of MeanAveragePrecision()
    """
    model.eval()
    with torch.no_grad():
        for i, (image_batch,targets) in enumerate(dataloader):
            update_metric_on_batch(metric,model,image_batch,targets)

    return metric
    
def evaluate_dataloader(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader) -> dict : 
    """
    model: A PyTorch model of type torch.nn.Module returned from create_model.py
    dataloader: An instance of torch.data.utils.Dataloader

    Returns:
        mAP_dict: A dictionary containing the following key-values:
        map: (Tensor)
        map_small: (Tensor)
        map_medium:(Tensor)
        map_large: (Tensor)
        mar_1: (Tensor)
        mar_10: (Tensor)
        mar_100: (Tensor)
        mar_small: (Tensor)
        mar_medium: (Tensor)
        mar_large: (Tensor)
        map_50: (Tensor) (-1 if 0.5 not in the list of iou thresholds)
        map_75: (Tensor) (-1 if 0.75 not in the list of iou thresholds)
        map_per_class: (Tensor) (-1 if class metrics are disabled)
        mar_100_per_class: (Tensor) (-1 if class metrics are disabled)
    """
    metric = MeanAveragePrecision(iou_type = 'bbox', class_metrics = True)
    update_metric_on_dataloader(metric,model,dataloader)
    mAP_dict = metric.compute()
    return mAP_dict

def evaluate_batch(model:torch.nn.Module, image_batch:torch.Tensor, targets:list) -> dict :
    """
    model: A PyTorch model of type torch.nn.Module returned from create_model.py
    image_batch: A batch of images of type torch.Tensor
    targets: A list of dictionaries corresponding to per-image targets.
        Each dictionary would have the following keys:
        boxes: FloatTensor of shape (num_boxes, 4) containing num_boxes ground truth boxes of the format specified in the constructor. 
            By default, this method expects (xmin, ymin, xmax, ymax) in absolute image coordinates.
        labels: IntTensor of shape (num_boxes) containing 0-indexed ground truth classes for the boxes.

    Returns:
        mAP_dict: A dictionary containing the following key-values:
        map: (Tensor)
        map_small: (Tensor)
        map_medium:(Tensor)
        map_large: (Tensor)
        mar_1: (Tensor)
        mar_10: (Tensor)
        mar_100: (Tensor)
        mar_small: (Tensor)
        mar_medium: (Tensor)
        mar_large: (Tensor)
        map_50: (Tensor) (-1 if 0.5 not in the list of iou thresholds)
        map_75: (Tensor) (-1 if 0.75 not in the list of iou thresholds)
        map_per_class: (Tensor) (-1 if class metrics are disabled)
        mar_100_per_class: (Tensor) (-1 if class metrics are disabled)
    """

    metric = MeanAveragePrecision(iou_type = 'bbox', class_metrics = True)
    update_metric_on_batch(metric, model, image_batch, targets)
    mAP_dict = metric.compute()
    return mAP_dict
    

if __name__ == "__main__":
    
    from create_model import get_model_fasterrcnn

    model = get_model_fasterrcnn(num_classes=3)
    image_batch = torch.rand(2,3,224,224)

    targets = {
        'boxes':torch.tensor([[0,0,100,100],[100,100,200,200]]),
        'labels':torch.tensor([0,1])
        }

    mAP_dict = evaluate_batch(model,image_batch,[targets,targets])

    print(mAP_dict)