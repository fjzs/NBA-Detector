# The purpose of this module is to provide an interface to create different models
# Examples: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights


def get_model(model_name:str = "fasterrcnn", num_classes:int = 4, trainable_backbone_layers:int=5) -> torch.nn.Module:
    """Creates a movel specified by its name and the number of classes

    Args:
    - model_name: pick one of ["fasterrcnn" (VOC format), "retinanet" (VOC format)]
    - num_classes: 1 (background) + non-background classes to predict
    - trainable_backbone_layers: number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

    Returns:
        torch.nn.Module: torch model
    """
    available_models = ["fasterrcnn", "retinanet"]
    if (model_name is None) or (num_classes is None) or (trainable_backbone_layers is None):
        raise ValueError("Cannot receive an input as None type")
    if model_name not in available_models:
        raise ValueError(f"{model_name} not recognized, available models are: {available_models}")
    if num_classes <= 1:
        raise ValueError(f"num_classes is {num_classes}, it has to be >= 2")
    if not(0 <= trainable_backbone_layers <= 5):
        raise ValueError(f"trainable_backbone_layers has to be within [0,5], value passed was {trainable_backbone_layers}")

    model = None    
    if model_name == "fasterrcnn":
        model = get_model_fasterrcnn(num_classes, trainable_backbone_layers)
    elif model_name == "retinanet":
        model = get_model_retinanet(num_classes, trainable_backbone_layers)
    else:
        raise ValueError(f"model name {model_name} not recognized")
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params}")
    print(f"Total trainable params: {pytorch_total_trainable_params}")
    return model

def get_model_retinanet(num_classes:int, trainable_backbone_layers:int=5) -> torch.nn.Module:
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.
    """
    model = retinanet_resnet50_fpn_v2(
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers = trainable_backbone_layers
    )
    return model


def get_model_fasterrcnn(num_classes:int, trainable_backbone_layers:int=5) -> torch.nn.Module:
    """
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    Args:
    - num_classes: 1 (background) + non-background classes to predict
    - trainable_backbone_layers: number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. Default is 3.

    Returns:
        torch.nn.Module:
    """

    # Source: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#finetuning-from-a-pretrained-model
    # Source for region proposal network: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT",
                                                                 trainable_backbone_layers=trainable_backbone_layers)
    # replace the classifier with a new one, that has num_classes which is user-defined
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



if __name__ == "__main__":
    model = get_model("retinanet", num_classes=4, trainable_backbone_layers=5)
    print(model)
    x = 0
    
    


