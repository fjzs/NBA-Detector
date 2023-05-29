from collections import defaultdict
import torch
from tqdm import tqdm
import math
import sys
import numpy as np

from src.nba_detector.evaluate import evaluate_dataloader
#from evaluate import evaluate_dataloader

def collate_fn(batch):
    images = [] # list(image for image in images)
    labels = []
    for image, label in batch:
        # labels.append({k: torch.Tensor(v) for k,v in label.items()})
        # Some images have no boxes in the current dataset which results in error when passed to the model.
        # TODO we should remove these from our dataset so that this is not required.
        if len(label['labels']) == 0: continue
        images.append(image.float()/255.0)
        labels.append({
            'boxes': label['boxes'],
            'labels': label['labels'],
        })
        # labels.append(label)
    return images, labels


def train_one_epoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        device: torch.device
    ):
    """Train one epoch."""
    model = model.to(device)
    model.train()

    print_freq = 4
    logger_list = defaultdict(list) # for each key provides a list of values
    for i, (images, labels) in enumerate(trainloader):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
        
        optimizer.zero_grad()
        loss_dict = model(images, labels)
        loss = sum(loss for loss in loss_dict.values())
        if not math.isfinite(loss):
            print(f"Infinite loss: {loss.item()}. Terminated")
            sys.exit(1)
        loss.backward()
        optimizer.step()

        # Logging
        #if (i>0 and i % print_freq == 0) or (i == len(trainloader) - 1) :
            #print(f"Epoch {epoch}: Batch {i}: mean_loss={loss.item()}", {k: v.item() for k,v in loss_dict.items()})
            # print(f"Epoch {epoch}: Batch {i}:", loss.item())
        for k,v in loss_dict.items():
            logger_list[k].append(v.item())

    # Compute a single value in training 
    logger_single_value = defaultdict(list)
    for k in logger_list:
        values = logger_list[k]
        avg = np.mean(values)
        logger_single_value[k] = avg

    # Evaluate on validation dataset
    val_metrics = evaluate_dataloader(model, valloader, device)
    for key in val_metrics:
        logger_single_value['val_' + str(key)] = val_metrics[key]
    #logger_single_value["val_map"] = val_metrics["map"].item()
    #logger_single_value["val_loss"] = val_metrics["loss"]
    train_loss = logger_single_value['loss_classifier'] + logger_single_value['loss_box_reg'] + logger_single_value['loss_objectness'] + logger_single_value['loss_rpn_box_reg']
    val_loss = logger_single_value['val_loss']
    val_map = logger_single_value['val_map']
    print(f"\n\t\ttrain_loss: {round(train_loss,3)}, val_loss: {round(val_loss,3)}, val_map: {round(val_map,3)}")
    return logger_single_value

def train(model: torch.nn.Module,
          filepath_to_save: str, 
          trainset: torch.utils.data.Dataset,
          valset: torch.utils.data.Dataset,
          num_epochs: int = 1,
          batch_size:int = 8):
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1, 
        drop_last=False,
        collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=1, 
        drop_last=False,
        collate_fn=collate_fn
    )    

    assert filepath_to_save.endswith(".pth"), f"filepath_to_save has to end with .pth"

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.005)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger = defaultdict(list)
    best_valMAP = 0
    for i in tqdm(range(num_epochs)):
        epoch_logs = train_one_epoch(model, optimizer, trainloader, valloader, device)

        valMAP = epoch_logs['val_map']
        if valMAP > best_valMAP:
            best_valMAP = valMAP
            print(f"\t\t*** Best valMAP so far of {best_valMAP}, saving model...")
            torch.save(model.state_dict(), filepath_to_save)

        # lr_scheduler.step()
        # Update Logs
        for k,v in epoch_logs.items():
            if isinstance(v, list):
                logger[k].extend(v)
            else:
                logger[k].append(v)
    
    print("*** End of training")
    return logger