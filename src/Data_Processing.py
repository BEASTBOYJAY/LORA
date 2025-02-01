import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os


class ImageClassificationDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for image classification tasks.

    This class loads image data from a specified directory structure where subdirectories
    represent class labels. Images are transformed for use in neural network models.

    Attributes:
        transform (torchvision.transforms.Compose): Transformations applied to the images.
        class_to_idx (dict): Mapping from class names to integer indices.
        image_dir (str): Path to the directory containing image data.
        image_paths (list): List of all image file paths.
        labels (list): List of corresponding class labels.
    """

    def __init__(self, image_dir):
        """
        Initializes the dataset by reading image paths and labels.

        Args:
            image_dir (str): Directory path containing class-labeled subdirectories with images.
        """
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

        # Populate image paths and corresponding labels
        for idx, class_names in enumerate(os.listdir(self.image_dir)):
            class_path = os.path.join(self.image_dir, class_names)
            if os.path.isdir(class_path):
                self.class_to_idx[class_names] = idx
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(idx)

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load image and convert to RGB to handle grayscale images
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, label

        image = self.transform(image)
        return image, label


def split_dataset(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and testing subsets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        train_ratio (float): Ratio of the dataset used for training.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def get_dataloader(image_dir, batch_size=32, shuffle=True):
    """
    Creates data loaders for the training and testing datasets.

    Args:
        image_dir (str): Directory path containing image data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        tuple: A tuple containing the training DataLoader, testing DataLoader, and class mapping.
    """
    dataset = ImageClassificationDataset(image_dir=image_dir)
    train_dataset, test_dataset = split_dataset(dataset)

    # Create DataLoaders for training and testing datasets
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # Map class indices back to class names
    class_mapping = {v: k for k, v in dataset.class_to_idx.items()}

    return train_data_loader, test_data_loader, class_mapping
