import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from Data_Processing import get_dataloader
import json
import os


class TrainWithLoRA:
    def __init__(
        self,
        model,
        image_dir,
        lr,
        batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader, self.test_loader, self.class_mapping = get_dataloader(
            image_dir=image_dir, batch_size=batch_size
        )
        with open("./class_mapping.json", "w") as f:
            json.dump(self.class_mapping, f)

        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100.0 * correct / total

            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

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

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
            )
        print("Training completed.")
        os.makedirs("model", exist_ok=True)
        torch.save(self.model, "model/fine_tuned_model.pth")
