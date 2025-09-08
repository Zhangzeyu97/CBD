# BID Project - Blind Image Decomposition Framework

## Project Overview
BID (Blind Image Decomposition) is a deep learning framework for multi-domain image restoration, particularly focused on **real-scenario deraining in driving conditions**. The project implements various neural network architectures for removing multiple weather-related degradations including rain streaks, snow, haze, and raindrops from images.

## Architecture Type
This is a **conditional image-to-image translation** framework based on deep convolutional neural networks, with a focus on the **CBDNet (Controllable Blind Decomposition Network)** architecture for Task II.A (real-scenario deraining).

## Key Features
- Multi-domain degradation removal (rain, snow, haze, raindrops)
- Flexible model architectures (BIDeN baseline variants, CBDNet)
- Comprehensive training and testing pipeline
- Support for multiple loss functions (L1/L2, VGG perceptual, BCE, Dice, Huber)
- Mixed precision training with AMP (Automatic Mixed Precision)
- Extensive evaluation metrics (PSNR, SSIM)

## Project Structure

### Core Directories
- **`data/`**: Dataset loaders and preprocessing modules
- **`models/`**: Model architectures and loss functions
- **`options/`**: Configuration and argument parsing
- **`util/`**: Utility functions for visualization, HTML generation, and data processing
- **`metrics/`**: MATLAB scripts for evaluation metrics

### Excluded Directories (contain trained models and datasets)
- `checkpoints/`: Saved model checkpoints
- `results-task2a/`: Test results and outputs
- `datasets/`: Raw dataset files

## Main Components

### 1. Models (`models/`)
The project implements multiple model variants:

#### Core Models:
- **`cbd_model.py`**: Main model for Task II.A - real-scenario deraining (our CBDNet)
  - Based on modified Restormer architecture
  - Uses AdamW optimizer with weight decay
  - Implements mixed precision training with GradScaler
  - Supports multiple degradation types (B=rain, C=snow, D=haze, E=raindrop)
  
- **`biden*_model.py`** (biden2-biden8): Various BIDeN (Blind Image Decomposition Network) baseline model variants from prior work
- **`raina_model.py`, rainb_model.py`**: Rain-specific models
- **`jointremoval_model.py`**: Joint degradation removal model

#### Key Components:
- **`base_model.py`**: Abstract base class for all models
- **`networks.py`**: Network building blocks (generators, discriminators, normalization layers)
- **`losses.py`**: Custom loss functions including VGG perceptual loss

### 2. Data Pipeline (`data/`)

#### Dataset Classes:
- **`raina_dataset.py`**: Main dataset for Task II.A
  - Loads multiple degradation types (A-I folders)
  - A: CityScape clean images
  - B: Rain streak masks
  - C: Snow masks
  - D1/D2/D3: Haze transmission maps (light/medium/heavy)
  - E1/E2: Raindrop masks
  - H, I: Additional degradation types

- **`unaligned*_dataset.py`**: Unpaired image-to-image translation datasets
- **`base_dataset.py`**: Base dataset class with common preprocessing

### 3. Training & Testing

#### Entry Points:
- **`train.py`**: Main training script
  - Supports multi-GPU training
  - Implements gradient accumulation
  - Automatic checkpoint saving
  - Visualization with Visdom

- **`test.py`**: Testing/inference script
  - Batch processing of test images
  - HTML result generation
  - Support for different input combinations

#### Shell Scripts:
- **`run.sh`**: Example training command for Task II.A
- **`test_all.sh`**: Runs all 6 test cases with different degradation combinations

### 4. Configuration (`options/`)
- **`base_options.py`**: Common options for training and testing
- **`train_options.py`**: Training-specific options (learning rate, epochs, etc.)
- **`test_options.py`**: Test-specific options

### 5. Evaluation
- **`psnr.py`**: Python script for computing PSNR and SSIM metrics
- **`metrics/`**: MATLAB evaluation scripts for additional metrics

## Training Configuration

### Default Hyperparameters:
```python
- Learning Rate: 0.00015 (AdamW optimizer)
- Batch Size: 1
- Image Size: 192x192 (crop size)
- Loss Weights:
  - L1/L2: 50.0
  - VGG: 15.0
  - BCE: 1.0
  - Dice: 2.0
  - Huber: 10.0
```

### Training Command Example:
```bash
python train.py --dataroot ./datasets/raina --name task2a --model cbd \
                --dataset_mode raina --lr 0.00015 --n_epochs 15
```

## Testing Scenarios

The framework supports 6 different test cases:
1. **Case 1**: Rain streak removal only (B)
2. **Case 2**: Rain + Snow removal (BC)
3. **Case 3**: Rain + Light haze (BD, intensity=0)
4. **Case 4**: Rain + Heavy haze (BD, intensity=2)
5. **Case 5**: Rain + Moderate haze + Raindrop (BDE, intensity=1)
6. **Case 6**: All degradations (BCDE, intensity=1)

## Key Technical Features

### 1. Mixed Precision Training
- Uses PyTorch's Automatic Mixed Precision (AMP)
- GradScaler for stable gradient computation
- Improved training speed and memory efficiency

### 2. Multi-Scale Loss Functions
- Combines pixel-level (L1/L2) and perceptual (VGG) losses
- Segmentation losses (BCE, Dice) for mask prediction
- PSNR loss for direct optimization

### 3. Flexible Data Loading
- Support for paired and unpaired datasets
- Dynamic degradation combination during training
- Configurable data augmentation

### 4. Comprehensive Visualization
- Real-time training monitoring with Visdom
- HTML result pages for qualitative evaluation
- Automatic metric computation and logging

## Usage Workflow

1. **Prepare Dataset**: Place data in appropriate folder structure under `datasets/`
2. **Configure Training**: Modify options in command line or options files
3. **Train Model**: Run `train.py` with desired configuration
4. **Test Model**: Use `test.py` for inference on test images
5. **Evaluate Results**: Run `psnr.py` for quantitative metrics

## Notes
- The project focuses on autonomous driving scenarios with weather-related degradations
- Supports both synthetic and real-world degradation patterns
- Extensible architecture allows easy addition of new models and losses
- Checkpointing system enables training resumption and model selection