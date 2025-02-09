# ResNet Fine-tuning with LoRA

A PyTorch implementation of ResNet model fine-tuning using Low-Rank Adaptation (LoRA). This project provides a flexible and efficient way to adapt pre-trained ResNet models (ResNet18 or ResNet50) for custom image classification tasks while maintaining memory efficiency.

## Features

- Support for ResNet18 and ResNet50 architectures
- Implementation of Low-Rank Adaptation (LoRA) for efficient fine-tuning
- Customizable training pipeline with YAML configuration
- Easy-to-use inference pipeline
- Data processing utilities for image classification tasks
- Progress tracking with tqdm
- Support for both CPU and CUDA training

## Project Structure

```
.
├── src/
│   ├── LORA.py              # LoRA implementation
│   ├── Data_Processing.py   # Dataset and dataloader utilities
│   ├── Inference_pipeline.py # Inference implementation
│   ├── load_model.py        # Model loading and modification
│   └── trainer.py           # Training pipeline
├── config.yaml              # Configuration file
├── train.py                # Training script
├── inference.py            # Inference script
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BEASTBOYJAY/LORA.git
cd LORA
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`config.yaml`) to manage training and inference parameters:

```yaml
Training:
  image_dir: "dataset"      # Dataset directory
  model_name: "resnet18"    # Model type (resnet18 or resnet50)
  num_classes: 1000         # Number of output classes
  learning_rate: 0.0005     # Learning rate
  epochs: 5                 # Number of training epochs
  batch_size: 32           # Batch size

Lora_config:    
  rank: 5                  # LoRA rank parameter
  alpha: 0.5               # LoRA scaling factor

Inference:    
  model_path: ""          # Path to saved model
  image_path: ""          # Path to input image
```

## Dataset Structure

Organize your dataset in the following structure:
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Usage

### Training

1. Modify the `config.yaml` file to set your desired parameters.
2. Run the training script:
```bash
python train.py
```

The training script will:
- Load and modify the specified ResNet model with LoRA
- Split your dataset into training and testing sets
- Train the model for the specified number of epochs
- Save the trained model to `model/fine_tuned_model.pth`
- Save the class mapping to `class_mapping.json`

### Inference

1. Update the `config.yaml` file with your model and image paths.
2. Run the inference script:
```bash
python inference.py
```

## Implementation Details

### LoRA (Low-Rank Adaptation)

The `LORA.py` module implements Low-Rank Adaptation, which adds trainable rank decomposition matrices to the original model layers. This approach allows for efficient fine-tuning with fewer parameters while maintaining model performance.

### Data Processing

The `Data_Processing.py` module handles:
- Custom dataset creation
- Image transformations and normalization
- Dataset splitting
- DataLoader creation

### Model Loading

The `load_model.py` module manages:
- Loading pre-trained ResNet models
- Modifying layers for LoRA adaptation
- Handling model architecture selection

### Training Pipeline

The `trainer.py` module provides:
- Training loop implementation
- Progress tracking
- Model evaluation
- Checkpointing

### Inference Pipeline

The `Inference_pipeline.py` module handles:
- Model loading
- Image preprocessing
- Prediction generation


## Acknowledgments

- The LoRA implementation is based on the paper "LoRA: Low-Rank Adaptation of Large Language Models"
- ResNet implementations from torchvision