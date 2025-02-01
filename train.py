from trainer import TrainWithLoRA
from src.load_model import LoadModel
import yaml


class Train:
    def __init__(self, config_path: str):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.model_name = config["Training"]["model_name"]
        self.num_classes = config["Training"]["num_classes"]
        self.image_dir = config["Training"]["image_dir"]
        self.lr = (
            config["Training"]["learning_rate"]
            if config["Training"]["learning_rate"] is not None
            else 0.001
        )
        self.epochs = (
            config["Training"]["epochs"]
            if config["Training"]["epochs"] is not None
            else 5
        )
        self.batch_size = (
            config["Training"]["batch_size"]
            if config["Training"]["batch_size"] is not None
            else 32
        )
        self.rank = (
            config["Lora_config"]["rank"]
            if config["Lora_config"]["rank"] is not None
            else 5
        )
        self.alpha = (
            config["Lora_config"]["alpha"]
            if config["Lora_config"]["alpha"] is not None
            else 0.5
        )

    def run(self):
        model = LoadModel(self.model_name, self.num_classes).run()
        trainer = TrainWithLoRA(model, self.image_dir, self.lr, self.batch_size)
        trainer.train(self.epochs)


if __name__ == "__main__":
    pipeline = Train("config.yaml")
    pipeline.run()
