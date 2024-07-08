import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# Define the CustomViTModel class
class CustomViTModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomViTModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-384")

        num_features = self.vit.config.hidden_size
        self.vit.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        x = outputs.logits
        x = F.adaptive_avg_pool2d(x.unsqueeze(-1).unsqueeze(-1), (1, 1)).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, image_processor):
    image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs['pixel_values']

# Function to predict the class and confidence
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item()

# Main execution
if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the image processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")

    # Load the trained model
    num_classes = 4
    model = CustomViTModel(num_classes)
    model.load_state_dict(torch.load('final_ms_model_4classes_vit_base_384_full_finetune.pth', map_location=device))
    model.to(device)

    # Define class names
    class_names = ['Control-Axial', 'Control-Sagittal', 'MS-Axial', 'MS-Sagittal']

    # Path to the MRI image you want to classify
    image_path = r"C:\Works\Data\projects\Python\Projects\MS\set2\Multiple Sclerosis\balanced\4C\Control-Sagittal\C-S (10).png"  # Replace with the actual path

    # Load and preprocess the image
    image_tensor = load_and_preprocess_image(image_path, image_processor)

    # Make prediction
    predicted_class_idx, confidence = predict(model, image_tensor, device)

    # Get the predicted class name
    predicted_class = class_names[predicted_class_idx]

    # Print results
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")