# The purpose of this module is to provide an interface to create different models
# Examples: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb

import torch
import torchvision
import torchinfo
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(model_name:str, num_classes:int, trainable_backbone_layers:int=3) -> torch.nn.Module:
    """Creates a movel specified by its name and the number of classes

    Args:
    - model_name: pick one of ["fasterrcnn" (VOC format)]
    - num_classes: 1 (background) + non-background classes to predict
    - trainable_backbone_layers: number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. Default is 3.

    Returns:
        torch.nn.Module: torch model
    """
    available_models = ["fasterrcnn"]
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
    
    return model


def get_model_fasterrcnn(num_classes:int, trainable_backbone_layers:int=5) -> torch.nn.Module:
    """Creates the fasterrcnn model, which expects VOC format

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

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {pytorch_total_params}")
    print(f"Total trainable params: {pytorch_total_trainable_params}")
    # torchinfo.summary(model, input_size=(1, 1, 1200, 1600))
    return model



#if __name__ == "__main__":
#    model = get_model("fasterrcnn", 5, 1)
    
    


