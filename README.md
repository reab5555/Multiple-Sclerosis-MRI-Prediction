<img src="appendix/icon.jpeg" width="150" alt="alt text">

# Multiple Sclerosis MRI Classification

## Project Overview
This repository contains the implementation of a Vision Transformer (ViT) model designed to predict Multiple Sclerosis (MS) from Magnetic Resonance Imaging (MRI) scans. The goal is to classify images into one of four categories: Control-Axial, Control-Sagittal, MS-Axial, and MS-Sagittal. 

By implementing the Vision Transformer (ViT) model for MRI scans and Multiple Sclerosis detection, we showcase the model's effectiveness in medical imaging applications.

## What is Multiple Sclerosis?
Multiple Sclerosis (MS) is a chronic demyelinating disease characterized by the presence of plaques in the white matter of the central nervous system. These plaques can disrupt the flow of information within the brain, and between the brain and the rest of the body. MS can be diagnosed using MRI, which helps identify the characteristic lesions associated with the disease. Usual onset	is around age 20–50. 

It can cause a wide range of symptoms including fatigue, vision problems, mobility issues, cognitive changes, numbness, pain, speech difficulties, mood changes, heat sensitivity, and muscle spasticity, which vary in severity and occurrence among individuals.

<img src="appendix/applsci-12-04920-g001.png" width="450" alt="alt text">

## Dataset
The dataset and initial modeling attempts are derived from the work by Macin et al. (2022). They proposed a computationally efficient model using Exemplar Multiple Parameters Local Phase Quantization (ExMPLPQ) combined with a k-nearest neighbor (kNN) classifier. Their model achieved high accuracy in detecting MS from MRI images.

**Original Article:** [An Accurate Multiple Sclerosis Detection Model Based on Exemplar Multiple Parameters Local Phase Quantization: ExMPLPQ](https://www.mdpi.com/2076-3417/12/10/4920)

The dataset consists of axial and sagittal MRI images divided into four classes:
1. **Control-Axial:** Axial MRI images from healthy individuals.
2. **Control-Sagittal:** Sagittal MRI images from healthy individuals.
3. **MS-Axial:** Axial MRI images from individuals diagnosed with MS.
4. **MS-Sagittal:** Sagittal MRI images from individuals diagnosed with MS.

### MRI Samples for Control and MS
<p align="left">
<img src="appendix/C-A (24).png" width="200" alt="alt text">
<img src="appendix/C-A (90).png" width="200" alt="alt text">
<img src="appendix/C-S (13).png" width="200" alt="alt text">
<img src="appendix/C-S (79).png" width="200" alt="alt text">
<img src="appendix/MS-A (124).png" width="200" alt="alt text">
<img src="appendix/MS-A (25).png" width="200" alt="alt text">
<img src="appendix/MS-S (101).png" width="200" alt="alt text">
<img src="appendix/MS-S (35).png" width="200" alt="alt text">
<p/>

### Sample Distribution
After we balanced the classes using downsampling, each class contains 646 samples (MRI images), ensuring a balanced dataset for training and evaluation.

## Model Architecture
We utilized the Vision Transformer (ViT) model, specifically `google/vit-base-patch16-384`. The core architecture and pre-trained weights of the ViT model were retained, while the final classification layers were fine-tuned on our dataset.

### Vision Transformer (ViT)
- **Patch Embeddings:** The input image is divided into fixed-size patches (e.g., 16x16), each of which is flattened and embedded into a vector.
- **Positional Encoding:** Added to the patch embeddings to maintain spatial information.
- **Transformer Encoder:** Processes the encoded patches through multi-head self-attention layers and feed-forward neural networks.
- **Classification Head:** The output is pooled and passed through a fully connected layer to generate final class probabilities.

### Detailed Architecture
- **Base Vision Transformer:**
  - Pre-trained on ImageNet-21k.
  - Added Adaptive Average Pooling to minimize overfitting.
  - **Fully Connected Layers:**
    - Connected Layer: 64 neurons, ReLU activation.
    - Linear layer with 4 output neurons (corresponding to our four classes).
   
The following table lists the hyperparameters used in our model:

| Hyperparameter   | Value           |
|------------------|-----------------|
| Input Size       | 1222x638        |  
| Dropout Rate     | 0.5             |
| Weight Decay     | 0.01            |
| Learning Rate    | 1e-5            |
| Optimizer        | AdamW           |
| Batch Size       | 32              |
| Patience         | 1               |

## Cross-Validation
To ensure robust evaluation and to prevent overfitting, we employed 6 fold cross-validation. This approach divides the dataset into six subsets, training the model on five subsets while using the sixth for validation (around 108 samples per fold-validation set).
  
## Results
Below are the results for the last epochs of each fold:

### Fold-wise Results
| Fold | Epochs | Train Loss | Val Loss | Val F1  | Accuracy | Precision | Recall |
|------|--------|------------|----------|---------|----------|-----------|--------|
| 1    | 10     | 0.0907     | 0.2323   | 0.9323  | 0.93     | 0.94      | 0.93   |
| 2    | 9      | 0.1007     | 0.1824   | 0.9513  | 0.95     | 0.95      | 0.95   |
| 3    | 9      | 0.0961     | 0.1740   | 0.9489  | 0.95     | 0.95      | 0.95   |
| 4    | 6      | 0.1852     | 0.2576   | 0.9233  | 0.92     | 0.93      | 0.92   |
| 5    | 8      | 0.1307     | 0.1447   | 0.9628  | 0.96     | 0.96      | 0.96   |
| 6    | 9      | 0.1108     | 0.1928   | 0.9418  | 0.94     | 0.94      | 0.94   |

### Class-wise Average Metrics
| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Control-Axial   | 0.9359    | 0.9675 | 0.9513   |
| Control-Sagittal| 0.9031    | 0.9799 | 0.9390   |
| MS-Axial        | 0.9683    | 0.9335 | 0.9504   |
| MS-Sagittal     | 0.9789    | 0.8934 | 0.9330   |

### Average Metrics across all Folds
| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 0.9435  |
| Precision       | 0.9464  |
| Recall          | 0.9282  |
| F1 Score        | 0.9434  |

<img src="appendix/learning_curve.png" width="400" alt="alt text">
<img src="appendix/confusion_matrix.png" width="400" alt="alt text">
  
## Conclusion
In conclusion, our project aimed to build upon the existing research by applying modern computer vision techniques, primarily the Vision Transformer model, to classify MS from MRI images. Through cross-validation and fine-tuning, we achieved robust performance metrics, demonstrating the efficacy of ViT models in medical image classification tasks.

## Potential Problems and Future Directions
### Small Dataset
- The current dataset, while balanced and sufficient for initial trials, is relatively small. Small datasets may not capture the full variability required for robust model training and generalization.

### Higher Resolution Images
- Higher resolution images could provide more detailed information, potentially improving the model's ability to detect subtle features associated with MS.

### Model Complexity and Interpretability
- Vision Transformers are complex models, which makes interpreting the learned features challenging. Understanding why the model makes specific predictions is crucial in medical applications.

## Acknowledgments
This project was made possible by the pioneering efforts of [Macin et al., 2022](https://www.mdpi.com/2076-3417/12/10/4920), whose data and initial study laid the groundwork for our research.

**References:**
Macin, G., Tasci, B., Tasci, I., Faust, O., Barua, P.D., Dogan, S., Tuncer, T., Tan, R.-S., Acharya, U.R. An Accurate Multiple Sclerosis Detection Model Based on Exemplar Multiple Parameters Local Phase Quantization: ExMPLPQ. Appl. Sci. 2022, 12, 4920.

---

**Disclaimer:**
This project is intended for research and educational purposes only and should not be used as a diagnostic tool without formal clinical validation.
