import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from .Data_Processing import get_dataloader
import json
import os


class TrainWithLoRA:
    """
    A class to train models with Low-Rank Adaptation (LoRA) for image classification.

    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        class_mapping (dict): Mapping of class indices to class names.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion (nn.Module): Loss function used for training.
        device (str): Device on which the model will be trained.
    """

    def __init__(
        self,
        model,
        image_dir,
        lr,
        batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initializes the training pipeline with data, model, and training parameters.

        Args:
            model (torch.nn.Module): Model to be trained.
            image_dir (str): Directory containing training and testing images.
            lr (float): Learning rate for the optimizer.
            batch_size (int): Batch size for training and testing.
            device (str, optional): Device for training ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        self.device = device
        self.model = model.to(self.device)
        self.train_loader, self.test_loader, self.class_mapping = get_dataloader(
            image_dir=image_dir, batch_size=batch_size
        )

        # Save class mapping to a JSON file
        with open("./class_mapping.json", "w") as f:
            json.dump(self.class_mapping, f)

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        """
        Trains the model and evaluates it on the test dataset.

        Args:
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Reset gradients for optimizer
                self.optimizer.zero_grad()

                # Forward pass and compute loss
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backpropagation and optimizer step
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100.0 * correct / total

            self.model.eval()  # Set model to evaluation mode
            test_loss = 0.0
            correct = 0
            total = 0

            # Test the model without computing gradients
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_loss /= len(self.test_loader)
            test_accuracy = 100.0 * correct / total

            # Print epoch summary
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
            )

        print("Training completed.")

        # Save the fine-tuned model
        os.makedirs("model", exist_ok=True)
        torch.save(self.model, "model/fine_tuned_model.pth")
