# Multiple Sclerosis MRI Prediction

## Project Overview
This repository contains the implementation of a Vision Transformer (ViT) model designed to predict Multiple Sclerosis (MS) from Magnetic Resonance Imaging (MRI) scans. The goal is to classify images into one of four categories: Control-Axial, Control-Sagittal, MS-Axial, and MS-Sagittal.

## What is Multiple Sclerosis?
Multiple Sclerosis (MS) is a chronic demyelinating disease characterized by the presence of plaques in the white matter of the central nervous system. These plaques can disrupt the flow of information within the brain, and between the brain and the rest of the body. MS can be diagnosed using MRI, which helps identify the characteristic lesions associated with the disease.

## Dataset Classes
The dataset used in this project was acquired from a previous study by [Macin et al., 2022](https://www.mdpi.com/2076-3417/12/10/4920).  

The dataset consists of axial and sagittal MRI images divided into four classes:
1. **Control-Axial:** Axial MRI images from healthy individuals.
2. **Control-Sagittal:** Sagittal MRI images from healthy individuals.
3. **MS-Axial:** Axial MRI images from individuals diagnosed with MS.
4. **MS-Sagittal:** Sagittal MRI images from individuals diagnosed with MS.

### Sample Distribution
Each class contains 750 samples, ensuring a balanced dataset for training and evaluation.

## Cross-Validation
To ensure robust evaluation and to prevent overfitting, we employed 6-fold cross-validation. This approach divides the dataset into six subsets, training the model on five subsets while using the sixth for validation. This process is repeated six times, with each subset used once for validation.

## Hyperparameters
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
| Device           | GPU if available|

## Model Architecture
We utilized the Vision Transformer (ViT) model, specifically `ViTForImageClassification` from the Hugging Face library. The core architecture and pre-trained weights of the ViT model were retained, while the final classification layers were fine-tuned on our dataset.

### Detailed Architecture
- **Base Vision Transformer:**
  - Pre-trained on ImageNet-21k.
  - Parameters were frozen to retain learned features.
- **Custom Classification Head:**
  - **Dropout Layer:** Dropout probability of 0.25 to prevent overfitting.
  - **Fully Connected Layers:**
    - First Layer: 1024 neurons, ReLU activation.
    - Second Layer: 512 neurons, ReLU activation.
    - Third Layer: 256 neurons, ReLU activation.
  - **Final Layer:**
    - Linear layer with 4 output neurons (corresponding to our four classes).

### Training Strategy
We utilized the AdamW optimizer due to its effective weight decay and learning rate scheduling capabilities. The learning rate scheduler (`ReduceLROnPlateau`) monitored validation loss, reducing the learning rate when validation performance plateaued.

## Results
### Cross-Validation Details
We conducted training and evaluation across six folds. Below are the results for the last epochs of each fold and the average metrics across all folds:

### Fold-wise Results
| Fold | Epoch | Train Loss | Val Loss | Val F1  | Accuracy | Precision | Recall |
|------|-------|------------|----------|---------|----------|-----------|--------|
| 1    | 7     | 0.3410     | 0.3753   | 0.8290  | 0.8300   | 0.8306    | 0.8200 |
| 2    | 5     | 0.4011     | 0.3873   | 0.8304  | 0.8300   | 0.8405    | 0.8300 |
| 3    | 8     | 0.3333     | 0.4006   | 0.8222  | 0.8200   | 0.8234    | 0.8156 |
| 4    | 11    | 0.3135     | 0.2977   | 0.8750  | 0.8800   | 0.8735    | 0.8800 |
| 5    | 5     | 0.4039     | 0.4531   | 0.7967  | 0.8000   | 0.8031    | 0.8000 |
| 6    | 9     | 0.3273     | 0.3481   | 0.8227  | 0.8250   | 0.8257    | 0.8231 |

### Average Metrics across all Folds
| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 0.8319  |
| F1 Score        | 0.8293  |

### Class-wise Average Metrics
| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Control-Axial   | 0.8438    | 0.9033 | 0.8681   |
| Control-Sagittal| 0.8069    | 0.8216 | 0.7978   |
| MS-Axial        | 0.9087    | 0.8203 | 0.8541   |
| MS-Sagittal     | 0.8410    | 0.7835 | 0.7975   |

## Model Fine-Tuning
In this project, we fine-tuned only the final layers of the Vision Transformer, retaining the majority of the base model's pre-trained weights. This approach allowed us to leverage the robust feature representations learned from the large-scale ImageNet dataset while adapting the classifier to our specific task.

## Related Work and Dataset
The dataset and initial modeling attempts are derived from the work by Macin et al. (2022). They proposed a computationally efficient model using Exemplar Multiple Parameters Local Phase Quantization (ExMPLPQ) combined with a k-nearest neighbor (kNN) classifier. Their model achieved high accuracy in detecting MS from MRI images.

- **Original Article:** [An Accurate Multiple Sclerosis Detection Model Based on Exemplar Multiple Parameters Local Phase Quantization: ExMPLPQ](https://www.mdpi.com/2076-3417/12/10/4920)

### Methodology Overview from Article
**Dataset Description:**
- MRI images were sourced from 72 MS patients and 59 healthy individuals.
- Original dataset size: 3427 images (1652 axial, 1775 sagittal).
- Ethical approval was obtained for the study, and images were read by medical experts.

**Model Description:**
- Utilized LPQ for feature extraction, creating a feature vector from 3x3, 5x5, and 7x7 overlapping blocks.
- Used iterative neighborhood component analysis (INCA) for feature selection.
- Applied a k-nearest neighbor (kNN) algorithm for binary classification.

**Performance:**
- Achieved high classification accuracy rates using 10-fold cross-validation: 98.37% for axial, 97.75% for sagittal, and 98.22% for combined datasets.

The dataset is publicly available and can be accessed from [Kaggle](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis).

### Significant Findings
- The ExMPLPQ model demonstrated that a computationally lightweight approach could achieve high accuracy in MS detection.
- Their approach has potential clinical applications for high-throughput screening of brain MRI images in suspected MS cases.

## Conclusion
In conclusion, our project aimed to build upon the existing research by applying modern deep learning techniques, primarily the Vision Transformer model, to classify MS from MRI images. Through meticulous cross-validation and fine-tuning, we achieved robust performance metrics, demonstrating the efficacy of ViT models in medical image classification tasks.

## Acknowledgments
This project was made possible by the pioneering efforts of [Macin et al., 2022](https://www.mdpi.com/2076-3417/12/10/4920), whose data and initial study laid the groundwork for our research.

## How to Use
To replicate our results or extend the work, follow these steps:
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/multiple-sclerosis-classification.git
   cd multiple-sclerosis-classification
   ```
2. **Install Dependencies:**
   Ensure you have the required packages. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the Dataset:**
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis) and place it in the appropriate directory.
4. **Run the Training Script:**
   To train the model, execute:
   ```bash
   python train.py
   ```
5. **Evaluate and Visualize Results:**
   The script will save the trained model and generate performance plots. You can review these to understand the model's behavior.

## Contact
For any inquiries or contributions, please reach out to [your.email@example.com](mailto:your.email@example.com).

---

**References:**
- Macin, G., Tasci, B., Tasci, I., Faust, O., Barua, P.D., Dogan, S., Tuncer, T., Tan, R.-S., Acharya, U.R. An Accurate Multiple Sclerosis Detection Model Based on Exemplar Multiple Parameters Local Phase Quantization: ExMPLPQ. Appl. Sci. 2022, 12, 4920.

---

**Disclaimer:**
This project is intended for research and educational purposes only and should not be used as a diagnostic tool without formal clinical validation.
