import torch
from torchvision import transforms
from PIL import Image
import json


class InferencePipeline:
    """
    A pipeline for running inference on an image classification model.

    This class supports LoRA (Low-Rank Adaptation) modules and handles image preprocessing,
    model loading, and prediction.

    Attributes:
        device (str): The device to run inference on (either 'cuda' or 'cpu').
        model (torch.nn.Module): The loaded model for inference.
        class_mapping (dict): Mapping from predicted indices to class names.
        transforms (torchvision.transforms.Compose): Image transformations for preprocessing.
    """

    def __init__(self, model_path, device=None):
        """
        Initializes the inference pipeline by loading the model and setting up transformations.

        Args:
            model_path (str): Path to the trained model file.
            lora_enable (bool): Whether to enable LoRA layers in the model.
            device (str, optional): Device to run the inference ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.class_mapping = json.load(open("./class_mapping.json", "r"))

        self.model.eval()
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path):
        """
        Predicts the class of an image.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            str: Predicted class label.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        input_data = self.transforms(image).unsqueeze(0)  # Add batch dimension

        # Perform inference without gradient computation
        with torch.no_grad():
            output = self.model(input_data.to(self.device))

        predicted_idx = torch.max(output, 1)[
            1
        ].item()  # Get the index of the max probability
        return self.class_mapping[predicted_idx]
