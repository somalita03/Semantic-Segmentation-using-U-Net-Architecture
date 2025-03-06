# Image Segmentation: U-Net For Self-Driving Cars

## Abstract

This project focuses on developing a semantic segmentation model using the U-Net architecture, specifically tailored for road scene images captured by in-car cameras. The primary objective is to accurately predict segmentation masks, a crucial component in autonomous driving systems. The project involved a thorough exploration of U-Net's encoder-decoder structure and the importance of skip connections for retaining spatial information.

Key steps included rigorous data preparation using the CamVid dataset, model training with iterative optimization, and tackling challenges such as class imbalance and overfitting. Strategies like regularization, early stopping, and data augmentation were implemented to enhance model generalization. The results demonstrated that the U-Net-based approach effectively segmented road scenes, identifying critical elements like lanes, vehicles, and pedestrians. This report highlights the project's milestones, challenges, and methodologies, paving the way for future real-time applications in autonomous driving.

## Introduction & Background

### Semantic Segmentation and U-Net

Semantic segmentation is a fundamental task in computer vision, crucial for applications requiring precise image analysis, such as autonomous driving. It classifies each pixel in an image into predefined categories, ensuring an accurate interpretation of road scenes. The U-Net architecture, originally developed for medical image segmentation, has proven effective for various pixel-level classification tasks due to its encoder-decoder structure with skip connections, which preserve spatial details while capturing high-level features.

### Internship Journey & Learning

Our internship began with an exploration of adversarial attacks and their impact on model performance. We studied the paper **ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**, which provided insights into robustness techniques against adversarial perturbations. Subsequently, we examined the **MedIS (Medical Image Segmentation) Attack** paper, which introduced adversarial strategies for medical image segmentation models. These studies informed our approach to model optimization and security considerations in semantic segmentation.

We implemented the U-Net model on the CamVid dataset, facing challenges such as image-mask mismatches, overfitting, and dataset inconsistencies. By employing normalization, hyperparameter tuning, and data augmentation, we iteratively improved performance, enhancing the model's accuracy and generalization.

## Problem Statement

### Challenges in Semantic Segmentation for Autonomous Driving

- **Pixel-Level Classification:** Ensuring accurate segmentation of road elements like lanes, vehicles, and pedestrians.
- **Handling Class Imbalance:** Addressing dataset issues where certain classes dominate, leading to biased predictions.
- **Overfitting & Generalization:** Improving the model's robustness to perform well across different scenarios.

## Key Solutions Implemented

- **Data Preprocessing:** Normalization, resizing, and augmentation to align images and masks properly.
- **Model Training & Evaluation:** Training U-Net with ReLU activation, Adam optimizer, and binary cross-entropy loss over 200 epochs.
- **Refinement & Optimization:** Hyperparameter tuning, dropout, and early stopping to prevent overfitting.
- **Advanced Analysis:** Class-wise accuracy evaluation, frequency analysis, and visualization using RGB color mapping.

## Current Approach

- **Data Preparation:** Normalizing and resizing the CamVid dataset, ensuring image-mask consistency.
- **Model Training:** Implementing U-Net in TensorFlow/Keras, training over 100 epochs with appropriate activation functions and optimizers.
- **One-Hot Encoding:** Transforming categorical data into binary matrices for better class differentiation.
- **Model Refinement:** Implementing techniques like data augmentation and hyperparameter adjustments.
- **Performance Evaluation:** Measuring accuracy, loss, and class-wise segmentation effectiveness.
- **Advanced Analysis:** Using histograms and visualizations for a detailed evaluation of model performance.

---

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install tensorflow keras albumentations matplotlib
   ```

2. **Prepare Dataset:**
   - Download the CamVid dataset.
   - Ensure image-mask pairs are correctly aligned.

3. **Train the Model:**
   ```bash
   python train.py
   ```

4. **Evaluate the Model:**
   ```bash
   python evaluate.py
   ```

5. **Visualize Predictions:**
   ```bash
   python visualize.py
   ```

## Future Work

- Improve real-time inference efficiency.
- Implement additional adversarial robustness techniques.
- Extend the model for multi-camera input in autonomous vehicles.

## Acknowledgments
Special thanks to our mentors and research community for their valuable insights and guidance in developing this project.
