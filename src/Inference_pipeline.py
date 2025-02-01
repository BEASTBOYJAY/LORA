import torch
from torchvision import transforms
from PIL import Image
from LORA import LoRA
import json


class Inference_pipeline:
    def __init__(
        self,
        model_path,
        lora_enable: bool = True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = torch.load(model_path, weights_only=False).to(self.device)
        self.class_mapping = json.load(open("./class_mapping.json", "r"))
        if lora_enable:
            for name, module in self.model.named_children():
                if isinstance(module, LoRA):
                    module.set_lora_enabled()
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
        image = Image.open(image_path).convert("RGB")
        input_data = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_data)
        predicted_idx = torch.max(output, 1)[1].item()
        return self.class_mapping[predicted_idx]
