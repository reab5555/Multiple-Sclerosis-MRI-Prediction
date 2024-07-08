import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, get_linear_schedule_with_warmup

device = "cuda"

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# Set the path to the main directory containing the class folders
main_folder = "mri_input_folder"
folders = [
    'Control-Axial',
    'Control-Sagittal',
    'MS-Axial',
    'MS-Sagittal'
]
folders = [os.path.join(main_folder, folder) for folder in folders]

n_folds = 6
batch_size = 32
lr = 1e-5
weight_decay = 0.01
patience = 1


def balance_dataset(images, labels):
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    print("Class counts before balancing:", class_counts)
    min_count = min(class_counts.values())
    print(f"Balancing to {min_count} samples per class")

    balanced_images = []
    balanced_labels = []
    class_counters = {label: 0 for label in class_counts}

    for image, label in zip(images, labels):
        if class_counters[label] < min_count:
            balanced_images.append(image)
            balanced_labels.append(label)
            class_counters[label] += 1

    print("Class counts after balancing:", {label: balanced_labels.count(label) for label in set(balanced_labels)})
    return balanced_images, balanced_labels


class CustomImageDataset(Dataset):
    def __init__(self, folders, processor):
        self.processor = processor
        self.images = []
        self.labels = []

        for i, folder in enumerate(folders):
            print(f"Processing folder: {folder}")
            folder_images = [os.path.join(folder, img_name) for img_name in os.listdir(folder) if
                             img_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  Found {len(folder_images)} images")
            self.images.extend(folder_images)
            self.labels.extend([i] * len(folder_images))

        unique_labels = set(self.labels)
        print(f"Unique labels before balancing: {unique_labels}")
        print(f"Total images before balancing: {len(self.images)}")

        # Balance the dataset
        self.images, self.labels = balance_dataset(self.images, self.labels)

        print(f"After balancing:")
        unique_labels = set(self.labels)
        for label in unique_labels:
            count = self.labels.count(label)
            print(f"  Class {label}: {count} images")

        print(f"Total images after balancing: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()
        label = self.labels[idx]
        return pixel_values, label

# Load the ViT model and image processor
model_name = "google/vit-base-patch16-384"
image_processor = ViTImageProcessor.from_pretrained(model_name)

class CustomViTModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomViTModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(model_name)

        # Replace the classifier
        num_features = self.vit.config.hidden_size
        self.vit.classifier = nn.Identity()  # Remove the original classifier

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        x = outputs.logits  # Use logits instead of last_hidden_state

        # Apply Global Average Pooling 2D
        x = F.adaptive_avg_pool2d(x.unsqueeze(-1).unsqueeze(-1), (1, 1)).squeeze(-1).squeeze(-1)

        # Pass through the classifier
        x = self.classifier(x)
        return x

def train_or_evaluate(model, loader, optimizer, criterion, device, is_training, scheduler=None):
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.set_grad_enabled(is_training):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, all_labels, all_preds, f1

# Create full dataset
full_dataset = CustomImageDataset(folders, image_processor)

# Determine the number of classes
num_classes = len(set(full_dataset.labels))
print(f"Number of classes detected: {num_classes}")

# Get class names
class_names = [os.path.basename(folder) for folder in folders]
print(f"Class names: {class_names}")

# Prepare for cross-validation
n_splits = n_folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

# Cross-validation loop
fold_results = []
all_val_labels = []
all_val_preds = []
all_train_losses = []
all_val_losses = []

for fold, (train_idx, val_idx) in enumerate(skf.split(full_dataset.images, full_dataset.labels), 1):
    print(f"Fold {fold}/{n_splits}")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_subsampler)
    val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_subsampler)

    model = CustomViTModel(num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_epochs = 30
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience = patience
    patience_counter = 0

    fold_train_losses = []
    fold_val_losses = []

    for epoch in range(num_epochs):
        train_loss, _, _, _ = train_or_evaluate(model, train_loader, optimizer, criterion, device, is_training=True, scheduler=scheduler)
        val_loss, val_labels, val_preds, val_f1 = train_or_evaluate(model, val_loader, optimizer, criterion, device, is_training=False)

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Store losses for this fold
    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Final evaluation
    _, val_labels, val_preds, val_f1 = train_or_evaluate(model, val_loader, optimizer, criterion, device, is_training=False)
    fold_results.append({
        'accuracy': (np.array(val_labels) == np.array(val_preds)).mean(),
        'f1': val_f1,
        'report': classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
    })
    all_val_labels.extend(val_labels)
    all_val_preds.extend(val_preds)

    print(f"Fold {fold} results:")
    print(classification_report(val_labels, val_preds, target_names=class_names))

# Calculate average metrics across folds
avg_results = {
    'accuracy': np.mean([fold['accuracy'] for fold in fold_results]),
    'f1': np.mean([fold['f1'] for fold in fold_results]),
    'report': {
        class_name: {
            metric: np.mean([fold['report'][class_name][metric] for fold in fold_results])
            for metric in ['precision', 'recall', 'f1-score']
        }
        for class_name in class_names
    }
}

print("\nAverage results across all folds:")
print(f"Accuracy: {avg_results['accuracy']:.4f}")
print(f"F1 Score: {avg_results['f1']:.4f}")
for class_name in class_names:
    print(f"\n{class_name}:")
    for metric, value in avg_results['report'][class_name].items():
        print(f"  {metric}: {value:.4f}")

# Confusion matrix (sum of all folds)
conf_matrix = confusion_matrix(all_val_labels, all_val_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Control-Axial', 'Control-Sagittal', 'MS-Axial', 'MS-Sagittal'],
            yticklabels=['Control-Axial', 'Control-Sagittal', 'MS-Axial', 'MS-Sagittal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (All Folds)')
plt.show()

# Save the final model
torch.save(model.state_dict(), 'final_ms_model_4classes_vit_base_384_full_finetune.pth')
print("Final model saved as 'final_ms_model_4classes_vit_base_384_full_finetune.pth'")