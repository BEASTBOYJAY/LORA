from src.Inference_pipeline import InferencePipeline
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

    def run(self):
        """
        Executes the inference pipeline to make predictions on the given image.
        """
        pipeline = InferencePipeline(self.model_path)
        result = pipeline.predict(self.image_path)
        print(result)


if __name__ == "__main__":
    pipeline = Inference("config.yaml")
    pipeline.run()
