from src.Inference_pipeline import Inference_pipeline
import yaml


class Inference:
    def __init__(self, config_path: str):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.model_path = config["Inference"]["model_path"]
        self.image_path = config["Inference"]["image_path"]
        self.lora_enable = (
            config["Inference"]["lora_enable"]
            if config["Inference"]["lora_enable"] is not None
            else True
        )

    def run(self):
        Inference_pipeline(self.model_path, self.lora_enable).predict(self.image_path)


if __name__ == "__main__":
    pipeline = Inference("config.yaml")
    pipeline.run()
