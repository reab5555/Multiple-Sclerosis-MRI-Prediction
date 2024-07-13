import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn.functional as F


# Define the CustomViTModel class (same as in the training script)
class CustomViTModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomViTModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-384")

        # Replace the classifier
        num_features = self.vit.config.hidden_size
        self.vit.classifier = nn.Identity()  # Remove the original classifier

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Single output for binary classification
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        x = outputs.logits
        x = F.adaptive_avg_pool2d(x.unsqueeze(-1).unsqueeze(-1), (1, 1)).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze()


# Function to load an image and preprocess it
def load_and_preprocess_image(image_path, processor):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((384, 384))
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].squeeze()


# Function to predict the probability
def predict_probability(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        probability = torch.sigmoid(output).item()
    return probability


# Main prediction script
if __name__ == "__main__":
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the saved model
    model = CustomViTModel()
    model.load_state_dict(torch.load('final_ms_model_2classes_vit_base_384_bce.pth', map_location=device))
    model.to(device)

    # Load the image processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")

    # Path to the image you want to predict
    image_path = r"image_input_path"  # Replace with the actual path to your image

    # Load and preprocess the image
    image_tensor = load_and_preprocess_image(image_path, image_processor)

    # Make prediction
    probability = predict_probability(model, image_tensor, device)

    # Print results
    print(f"Probability of MS: {probability:.4f}")
    print(f"Probability of NON-MS: {1 - probability:.4f}")

    if probability > 0.5:
        print("Prediction: MS")
    else:
        print("Prediction: NON-MS")