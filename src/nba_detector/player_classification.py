import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from dataset import download_dataset_from_roboflow
from player_dataset import get_player_team_dataset

# Carry out knn classification on coloured images where input is a list of images and list of labels. The output is a list of predicted labels.


def knn_preprocess(images: list, labels: list):
    # Convert images to grayscale
    images = [image.numpy().astype(
        np.uint8) for image in images]
    gray_images = [cv2.resize(image, (200, 200)) for image in images]
    # Convert images to 1D array
    gray_images = [image.flatten() for image in gray_images]
    # Convert labels to 1D array
    labels = np.array(labels)
    # Convert images to 2D array
    gray_images = np.array(gray_images)
    return gray_images, labels


def knn_classification(images: list, labels: list, k: int = 5):
    gray_images, labels = knn_preprocess(images, labels)
    # Create knn model
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit model
    knn.fit(gray_images, labels)
    return knn


def confusion_matrix(test_labels, team_predictions):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(test_labels, team_predictions)
    plt.figure(figsize=(8, 6))
    classes_test = ['GSW', 'MEM', 'MTW', 'SAS']
    classes_pred = ['GSW', 'MEM', 'MTW', 'SAS']
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes_test,
                yticklabels=classes_pred)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.xticks(ticks=np.arange(len(classes_test))+0.5, labels=classes_test)
    plt.yticks(ticks=np.arange(len(classes_pred))+0.5, labels=classes_pred)

    plt.show()


"""
version_id = 9
dataset_path: str = 'NBA-Player-Detector-{}'.format(version_id)
if dataset_path not in os.listdir():
    download_dataset_from_roboflow(format='coco', version_id=version_id)
cropped_images, labels = get_player_team_dataset(
    os.path.join(dataset_path, 'train'))
train_images, test_images, train_labels, test_labels = train_test_split(cropped_images, labels, test_size=0.01)
knn_model = knn_classification(train_images, train_labels, 4)
test_images, test_labels = knn_preprocess(test_images, test_labels)
team_predictions = knn_model.predict(test_images)
print(sum([test_labels[i] == team_predictions[i] for i in range(len(test_labels))]), len(test_labels))

test_cropped_images, test_labels = get_player_team_dataset(
    os.path.join(dataset_path, 'valid'))
test_cropped_images, test_labels = knn_preprocess(test_cropped_images, test_labels)
team_predictions = knn_model.predict(test_cropped_images)
print(sum([test_labels[i] == team_predictions[i] for i in range(len(test_labels))]), len(test_labels))

"""
