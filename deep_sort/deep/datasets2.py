import json
import os
import random

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ClsDataset(Dataset):
    def __init__(self, images_path, images_labels, transform=None):
        self.images_path = images_path
        self.images_labels = images_labels
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        label = self.images_labels[idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


import os
import json
import random
from matplotlib import pyplot as plt

def read_split_data(root, valid_rate=0.2):
    assert os.path.exists(root), 'dataset root: {} does not exist.'.format(root)

    train_images_path = []
    train_labels = []
    val_images_path = []
    val_labels = []
    per_class_num = {}

    supported = ['.jpg', '.JPG', '.png', '.PNG']
    
    train_folder_path = root
    if not os.path.isdir(train_folder_path):
        raise ValueError(f"Training folder {train_folder_path} does not exist.")

    for file_name in os.listdir(train_folder_path):
        if os.path.splitext(file_name)[-1] not in supported:
            continue
        
        class_id = file_name.split('_')[0]  # Get the class from the filename
        img_path = os.path.join(train_folder_path, file_name)
        
        if class_id not in per_class_num:
            per_class_num[class_id] = 0
        per_class_num[class_id] += 1

        val = random.random() < valid_rate
        if val:
            val_images_path.append(img_path)
            val_labels.append(class_id)
        else:
            train_images_path.append(img_path)
            train_labels.append(class_id)
    
    class_names = sorted(per_class_num.keys())
    class_indices = {name: i for i, name in enumerate(class_names)}

    train_labels = [class_indices[label] for label in train_labels]
    val_labels = [class_indices[label] for label in val_labels]

    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    print("{} images were found in the training dataset.".format(sum(per_class_num.values())))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    assert len(train_images_path) > 0, "number of training images must greater than zero"
    assert len(val_images_path) > 0, "number of validation images must greater than zero"

    plot_distribution = False
    if plot_distribution:
        plt.bar(range(len(class_names)), [per_class_num[cls] for cls in class_names], align='center')
        plt.xticks(range(len(class_names)), class_names)

        for i, v in enumerate([per_class_num[cls] for cls in class_names]):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('classes')
        plt.ylabel('numbers')
        plt.title('the distribution of dataset')
        plt.show()

    return [train_images_path, train_labels], [val_images_path, val_labels], len(class_names)
