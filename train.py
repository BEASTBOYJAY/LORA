from src.trainer import TrainWithLoRA
from src.load_model import LoadModel
import yaml


class Train:
    """
    A class used to configure and run a training pipeline with LoRA for fine-tuning models.

    Attributes:
        model_name (str): The name of the model to be trained.
        num_classes (int): The number of output classes for classification.
        image_dir (str): Directory path for training images.
        lr (float): Learning rate for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        rank (int): Rank of the LoRA low-rank adaptation.
        alpha (float): Scaling factor for LoRA.

    Methods:
        __init__(config_path: str):
            Initializes the training configuration from a YAML file.

        run():
            Loads the model, sets up the trainer, and starts training.
    """

    def __init__(self, config_path: str):
        """
        Initializes the Train class with configuration parameters from a YAML file.

        Args:
            config_path (str): Path to the configuration file (YAML format).

        Initializes:
            model_name (str): The model to be trained.
            num_classes (int): Number of classes for classification.
            image_dir (str): Directory containing training images.
            lr (float): Learning rate, default is 0.001 if not specified.
            epochs (int): Number of epochs, default is 5 if not specified.
            batch_size (int): Batch size, default is 32 if not specified.
            rank (int): LoRA rank, default is 5 if not specified.
            alpha (float): LoRA alpha value, default is 0.5 if not specified.
        """
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
        """
        Loads the model, sets up the LoRA-based trainer, and begins the training process.

        This method initializes the model using the provided configuration and then
        starts the training using the specified number of epochs, learning rate,
        and batch size.

        Runs:
            Loads the model and trains it using the specified configuration.
        """
        # Load the model with the required number of classes
        model = LoadModel(self.model_name, self.num_classes).run()

        # Initialize the LoRA-based trainer with the model, training data, and hyperparameters
        trainer = TrainWithLoRA(model, self.image_dir, self.lr, self.batch_size)

        # Start the training process for the specified number of epochs
        trainer.train(self.epochs)


if __name__ == "__main__":
    # Initialize the training pipeline with the provided configuration file
    pipeline = Train("config.yaml")

    # Run the training pipeline
    pipeline.run()
