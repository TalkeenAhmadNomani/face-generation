# Synthetic Face Generation and Classification using WGAN-GP and CNN

## Overview
This project implements a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** to generate synthetic celebrity face images and a **Convolutional Neural Network (CNN)** to classify them based on the "Young" attribute using the CelebA dataset (~202,599 images). Developed on Kaggle, leveraging GPU resources.

## Key Objectives
- Generate high-quality synthetic face images using WGAN-GP.
- Optimize WGAN-GP hyperparameters to improve image quality.
- Train a CNN classifier to predict the "Young" attribute.
- Evaluate the impact of synthetic data on classification performance.

## Dataset
- **Source:** CelebA dataset with 202,599 annotated celebrity face images.  
- **Sample used:** 30,000 images (random seed = 42)  
- **Split:**
  - Train: 24,102 (~80%)
  - Validation: 2,941 (~10%)
  - Test: 2,957 (~10%)  
- **Preprocessing:**  
  - Resize: (3 × 64 × 64)  
  - Normalize: mean = 0.5, std = 0.5  
  - "Young" attribute remapped to 0/1  

## Methodology

### Data Preparation
- Custom `CustomCelebADataset` class  
- PyTorch `DataLoader` with batch size = 128  
- Visual validation of image preprocessing  

### WGAN-GP Implementation
- **Generator:** 5 transposed convolutional layers, latent dim = 100, Tanh output  
- **Discriminator:** 5 convolutional layers, LeakyReLU, scalar output  
- **Training:** 30 epochs, Adam optimizer (lr = 1e-4, β₁ = 0, β₂ = 0.9), 5 critic updates per generator, gradient penalty = 10  
- **Evaluation:** Inception Score (IS) & Fréchet Inception Distance (FID) on 2,500 synthetic images  
- **Visualization:** Loss curves, image grids  

### Hyperparameter Tuning
- **Framework:** Optuna  
- Tuned: lr, batch size, latent dim, critic updates  
- **Best config:**
  - lr = 0.000414  
  - batch size = 64  
  - latent dim = 64  
  - critic updates = 9  

### CNN Classifier (Real Data)
- 3 convolutional layers → ReLU → MaxPool  
- 2 fully connected layers → ReLU → Dropout (0.5) → Sigmoid  
- Training: 30 epochs on real data  
- Optimizer: Adam (lr = 0.001), BCE loss  
- Model saved as `cnn_classifier_real_data.pth`  

### Synthetic Data Annotation
- Generated 16,871 synthetic images  
- Annotated via real-data-trained CNN (threshold = 0.5)  
- Saved: `synthetic_images.pt`, `synthetic_labels.pt`  

### CNN Classifier (Mixed Data)
- Mixed dataset: 33,742 images (50% real, 50% synthetic)  
- Retrained CNN for 30 epochs  
- Evaluated on test set  

## Results

### WGAN-GP Performance
| Model | IS | FID |
|-------|----|-----|
| Initial | 2.74 ± 0.12 | 585.03 |
| Tuned   | 3.26 ± 0.20 | 983.53 |

### CNN Classifier (Real Data)
- Accuracy: 0.8380  
- Precision: 0.8504  
- Recall: 0.9524  
- F1-Score: 0.8985  

### CNN Classifier (Mixed Data)
- Accuracy: 0.8157  
- Precision: 0.8564  
- Recall: 0.9075  
- F1-Score: 0.8812  

## Conclusion
This project demonstrates the generation of realistic synthetic faces using WGAN-GP and evaluates their utility for improving CNN classification performance. The integration of synthetic data shows a slight drop in accuracy but maintains high precision and recall, proving its potential for data augmentation in face attribute classification tasks.

## Tech Stack
- **Deep Learning:** PyTorch  
- **Generative Models:** WGAN-GP  
- **Classifier:** CNN  
- **Hyperparameter Tuning:** Optuna  
- **Platform:** Kaggle (GPU)
