from collections import defaultdict
import torch
from tqdm import tqdm
import math
import numpy as np
import sys

from src.nba_detector.evaluate import evaluate_dataloader


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
        device: torch.device, epoch: int
    ):
    """Train one epoch."""
    model = model.to(device)
    model.train()

    print_freq = 4
    logger = defaultdict(list)
    for i, (images, labels) in tqdm(enumerate(trainloader), desc=f"Epoch {epoch:02d}: Batches"):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
        # print([{k: v.shape for k,v in t.items()} for t in labels])

        optimizer.zero_grad()
        loss_dict = model(images, labels)
        loss = sum(loss for loss in loss_dict.values())
        if not math.isfinite(loss):
            print(f"Infinite loss: {loss.item()}. Terminated")
            sys.exit(1)

        loss.backward()
        optimizer.step()

        # Logging
        if (i>0 and i % print_freq == 0) or (i == len(trainloader) - 1) :
            print(f"Epoch {epoch}: Batch {i}: mean_loss={loss.item()}", {k: v.item() for k,v in loss_dict.items()})
            # print(f"Epoch {epoch}: Batch {i}:", loss.item())
        for k,v in loss_dict.items():
            logger[k].append(v.item())
        logger["train/loss"].append(loss.item())

    logger["train/loss"] = np.mean(logger["train/loss"])

    # Evaluate on validation dataset
    val_metrics = evaluate_dataloader(model, valloader, device)
    logger["val/map"] = val_metrics["map"]
    logger["val/loss"] = val_metrics["loss"]
    return logger


def train(model: torch.nn.Module, trainset: torch.utils.data.Dataset, valset: torch.utils.data.Dataset, config: dict):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2, drop_last=True,
        collate_fn=collate_fn
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=config['batch_size'], shuffle=False, num_workers=2, drop_last=False,
        collate_fn=collate_fn
    )
    print("Num training samples:", len(trainset))
    print("Num validation samples:", len(valset))
    print("Steps per Epoch:", len(trainloader))

    # Import wandb
    if config['use_wandb']: import wandb

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=float(config['learning_rate']), momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    logger = defaultdict(list)
    best_val_map = 0.0
    for i in range(config['epochs']):
        epoch_logs = train_one_epoch(model, optimizer, trainloader, valloader, device, i)
        # lr_scheduler.step()
        # Update Logs
        for k,v in epoch_logs.items():
            if isinstance(v, list):
                logger[k].extend(v)
            else:
                logger[k].append(v)

        # wandb logging
        if config['use_wandb']:
            for k,v in epoch_logs.items():
                if k.startswith('train') or k.startswith('val'):
                    wandb.log({k: v}, step=i)

        # Save best val model
        if epoch_logs["val/map"] > best_val_map:
            best_val_map = epoch_logs["val/map"]
            torch.save(model.state_dict(), config['save_model_as'] + "_best_val_map.pt")

    # Save model checkpoint
    torch.save(model.state_dict(), config['save_model_as'] + ".pt")
    # # Save checkpoint using wandb
    # if config['use_wandb']:
    #     wandb.save(config['save_model_as'] + ".pt")
    print("*** End of training")
    return logger