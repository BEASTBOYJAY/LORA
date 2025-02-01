import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import random_split


class ImageClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.class_to_idx = {}
        self.image_dir = image_dir
        self.image_paths = []
        self.labels = []

        for idx, class_names in enumerate(os.listdir(self.image_dir)):
            class_path = os.path.join(self.image_dir, class_names)
            if os.path.isdir(class_path):
                self.class_to_idx[class_names] = idx
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, label

        image = self.transform(image)
        return image, label


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))

    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def get_dataloader(image_dir, batch_size=32, shuffle=True):

    dataset = ImageClassificationDataset(image_dir=image_dir)
    train_dataset, test_dataset = split_dataset(dataset)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    class_mapping = {v: k for k, v in dataset.class_to_idx.items()}
    return train_data_loader, test_data_loader, class_mapping
