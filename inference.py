from src.Inference_pipeline import Inference_pipeline
import yaml


class Inference:
    """
    A class to manage model inference using configurations from a YAML file.

    Attributes:
        model_path (str): Path to the trained model.
        image_path (str): Path to the input image for inference.
        lora_enable (bool): Flag to enable or disable LoRA layers during inference.
    """

    def __init__(self, config_path: str):
        """
        Initializes the inference pipeline by loading configurations.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        self.model_path = config["Inference"].get(
            "model_path", "model/fine_tuned_model.pth"
        )
        self.image_path = config["Inference"].get("image_path", "input_image.jpg")
        self.lora_enable = config["Inference"].get("lora_enable", True)

    def run(self):
        """
        Executes the inference pipeline to make predictions on the given image.
        """
        pipeline = Inference_pipeline(self.model_path, self.lora_enable)
        pipeline.predict(self.image_path)


if __name__ == "__main__":
    pipeline = Inference("config.yaml")
    pipeline.run()
