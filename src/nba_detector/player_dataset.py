from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
from numpy import uint8
import os

from dataset import download_dataset_from_roboflow


class PlayerDataset(Dataset):
    def __init__(self, dataset_path: str, annotations_path: str, player_category_id: int, transform: transforms = None):
        super(PlayerDataset).__init__()
        self.dataset_path = dataset_path
        self.annotations_path = annotations_path
        self.transform = transform
        self.player_category_id = player_category_id
        self.coco_dataset = CocoDetection(
            root=dataset_path, annFile=annotations_path, transform=transform)

    def __getitem__(self, index):
        image, target = self.coco_dataset[index]
        image = (image.permute((1, 2, 0))*255).numpy().astype(uint8).copy()

        labels: list[int] = []
        cropped_images: list = []

        for ann in target:
            if 'extra' in ann and 'attributes' in ann['extra'] and ann['category_id'] == self.player_category_id:
                bbox: list[int] = list(map(int, ann['bbox']))
                labels.append(ann['extra']['attributes'])
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + \
                    bbox[2], bbox[1]+bbox[3]
                cropped_images.append(image[y1:y2+1, x1:x2+1])
        return cropped_images, labels

    def __len__(self):
        return len(self.coco_dataset)


def get_player_team_dataset(path: str, annotations_path: str = ''):
    annotations_path = os.path.join(path, '_annotations.coco.json') if len(
        annotations_path) == 0 else annotations_path
    transform = transforms.ToTensor()

    def collate(batch):
        batch = list(filter(lambda b: len(b[0]) > 0 and len(b[1]) > 0, batch))
        if len(batch) == 0:
            return None
        return default_collate(batch)

    player_dataset = PlayerDataset(path, os.path.join(
        path, '_annotations.coco.json'), 2, transform)
    player_dataloader = DataLoader(
        player_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    cropped_images, labels = [], []
    for item in player_dataloader:
        if item:
            ci, l = item
            for im, label in zip(ci, l):
                cropped_images.append(im[0])
                labels.append(label)
    return cropped_images, labels


download_dataset_from_roboflow(format = 'coco', version_id = 8)
cropped_images, labels = get_player_team_dataset('/NBA-Player-Detector-8/train')
for c, l in zip(cropped_images, labels):
    plt.imshow(c)
    print(l)
    plt.show()
