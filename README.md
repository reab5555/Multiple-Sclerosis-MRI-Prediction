<img src="icon.jpeg" width="150" alt="alt text">

# Multiple Sclerosis MRI Classification

## Project Overview
This repository contains the implementation of a Vision Transformer (ViT) model designed to predict Multiple Sclerosis (MS) from Magnetic Resonance Imaging (MRI) scans. The goal is to classify images into one of four categories: Control-Axial, Control-Sagittal, MS-Axial, and MS-Sagittal. 

By implementing the Vision Transformer (ViT) model for MRI scans and Multiple Sclerosis detection, we showcase the model's effectiveness in medical imaging applications.

## What is Multiple Sclerosis?
Multiple Sclerosis (MS) is a chronic demyelinating disease characterized by the presence of plaques in the white matter of the central nervous system. These plaques can disrupt the flow of information within the brain, and between the brain and the rest of the body. MS can be diagnosed using MRI, which helps identify the characteristic lesions associated with the disease. Usual onset	is around age 20–50.

<img src="applsci-12-04920-g001.png" width="450" alt="alt text">

## Dataset
The dataset and initial modeling attempts are derived from the work by Macin et al. (2022). They proposed a computationally efficient model using Exemplar Multiple Parameters Local Phase Quantization (ExMPLPQ) combined with a k-nearest neighbor (kNN) classifier. Their model achieved high accuracy in detecting MS from MRI images.

**Original Article:** [An Accurate Multiple Sclerosis Detection Model Based on Exemplar Multiple Parameters Local Phase Quantization: ExMPLPQ](https://www.mdpi.com/2076-3417/12/10/4920)

### Methodology Overview from Article
**Dataset Description:**
- MRI images were sourced from 72 MS patients and 59 healthy individuals.
- Original dataset size: 3427 images (1652 axial, 1775 sagittal).
- Ethical approval was obtained for the study, and images were read by medical experts.

The dataset consists of axial and sagittal MRI images divided into four classes:
1. **Control-Axial:** Axial MRI images from healthy individuals.
2. **Control-Sagittal:** Sagittal MRI images from healthy individuals.
3. **MS-Axial:** Axial MRI images from individuals diagnosed with MS.
4. **MS-Sagittal:** Sagittal MRI images from individuals diagnosed with MS.

### Sample Distribution
After we balanced the classes using downsampling, each class contains 750 samples (MRI images), ensuring a balanced dataset for training and evaluation.

## Model Architecture
We utilized the Vision Transformer (ViT) model, specifically `google/vit-base-patch16-384`. The core architecture and pre-trained weights of the ViT model were retained, while the final classification layers were fine-tuned on our dataset.

### How Vision Transformer (ViT) Works
- **Patch Embeddings:** The input image is divided into fixed-size patches (e.g., 16x16), each of which is flattened and embedded into a vector.
- **Positional Encoding:** Added to the patch embeddings to maintain spatial information.
- **Transformer Encoder:** Processes the encoded patches through multi-head self-attention layers and feed-forward neural networks.
- **Classification Head:** The output is pooled and passed through a fully connected layer to generate final class probabilities.

We fine-tuned only the final layers of the Vision Transformer, retaining the majority of the base model's pre-trained weights. This approach allowed us to leverage the robust feature representations learned from the large-scale ImageNet dataset while adapting the classifier to our specific task.

### Detailed Architecture
- **Base Vision Transformer:**
  - Pre-trained on ImageNet-21k.
  - Parameters were frozen to retain learned features.
  - **Fully Connected Layers:**
    - First Layer: 1024 neurons, ReLU activation.
    - Second Layer: 512 neurons, ReLU activation.
    - Third Layer: 256 neurons, ReLU activation.
  - **Final Layer:**
    - Linear layer with 4 output neurons (corresponding to our four classes).
   
The following table lists the hyperparameters used in our model:

| Hyperparameter   | Value           |
|------------------|-----------------|
| Dropout Rate     | 0.25            |
| Learning Rate    | 5e-5            |
| Optimizer        | AdamW           |
| Scheduler        | ReduceLROnPlateau|
| Epochs           | 30              |
| Batch Size       | 32              |
| Patience         | 1               |

## Cross-Validation
To ensure robust evaluation and to prevent overfitting, we employed 6-fold cross-validation. This approach divides the dataset into six subsets, training the model on five subsets while using the sixth for validation (around 100-110 samples per validation set).
  
## Results
Below are the results for the last epochs of each fold:

### Fold-wise Results
| Fold | Epochs | Train Loss | Val Loss | Val F1  | Accuracy | Precision | Recall |
|------|-------|------------|----------|---------|----------|-----------|--------|
| 1    | 7     | 0.3410     | 0.3753   | 0.8290  | 0.8300   | 0.8306    | 0.8200 |
| 2    | 5     | 0.4011     | 0.3873   | 0.8304  | 0.8300   | 0.8405    | 0.8300 |
| 3    | 8     | 0.3333     | 0.4006   | 0.8222  | 0.8200   | 0.8234    | 0.8156 |
| 4    | 11    | 0.3135     | 0.2977   | 0.8750  | 0.8800   | 0.8735    | 0.8800 |
| 5    | 5     | 0.4039     | 0.4531   | 0.7967  | 0.8000   | 0.8031    | 0.8000 |
| 6    | 9     | 0.3273     | 0.3481   | 0.8227  | 0.8250   | 0.8257    | 0.8231 |

### Class-wise Average Metrics
| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Control-Axial   | 0.8438    | 0.9033 | 0.8681   |
| Control-Sagittal| 0.8069    | 0.8216 | 0.7978   |
| MS-Axial        | 0.9087    | 0.8203 | 0.8541   |
| MS-Sagittal     | 0.8410    | 0.7835 | 0.7975   |

### Average Metrics across all Folds
| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 0.8319  |
| F1 Score        | 0.8293  |


<p align="left">
<img src="images/learning_curve.jpg" width="450" alt="alt text">
<img src="images/confusion_matrix.jpg" width="450" alt="alt text">
<p/>
  
## Conclusion
In conclusion, our project aimed to build upon the existing research by applying modern deep learning techniques, primarily the Vision Transformer model, to classify MS from MRI images. Through meticulous cross-validation and fine-tuning, we achieved robust performance metrics, demonstrating the efficacy of ViT models in medical image classification tasks.

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
