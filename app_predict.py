import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import plotly.graph_objects as go
from transformers import ViTImageProcessor, ViTForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the ViT model and image processor
model_name = "google/vit-base-patch16-384"
image_processor = ViTImageProcessor.from_pretrained(model_name)

class CustomViTModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomViTModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(model_name)
        num_features = self.vit.config.hidden_size
        self.vit.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        x = outputs.logits
        x = nn.functional.adaptive_avg_pool2d(x.unsqueeze(-1).unsqueeze(-1), (1, 1)).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze()

# Load the trained model
model = CustomViTModel()
model.load_state_dict(torch.load('final_ms_model_2classes_vit_base_384_bce.pth', map_location=device))
model.to(device)
model.eval()

def predict(image):
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img = img.resize((384, 384))
    inputs = image_processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        output = model(pixel_values)
        probability = torch.sigmoid(output).item()

    ms_prob = probability
    non_ms_prob = 1 - probability

    fig = go.Figure(data=[
        go.Bar(name='Non-MS', x=['Non-MS'], y=[non_ms_prob * 100], marker_color='blue'),
        go.Bar(name='MS', x=['MS'], y=[ms_prob * 100], marker_color='red')
    ])
    fig.update_layout(
        title='Prediction Probabilities',
        yaxis_title='Probability (%)',
        barmode='group',
        yaxis=dict(range=[0, 100])
    )

    prediction = "MS" if ms_prob > 0.5 else "Non-MS"
    confidence = max(ms_prob, non_ms_prob) * 100
    result_text = f"Prediction: {prediction}\nConfidence: {confidence:.2f}%"

    return result_text, fig

def load_readme():
    try:
        with open('README_DESC.md', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "README.md file not found. Please make sure it exists in the same directory as this script."

with gr.Blocks() as demo:
    gr.Markdown("# MS Prediction App")
    
    with gr.Tabs():
        with gr.TabItem("Prediction"):
            gr.Markdown("## Upload an MRI scan image to predict MS or Non-MS patient.")
            with gr.Row():
                input_image = gr.Image()
            predict_button = gr.Button("Predict")
            output_text = gr.Textbox()
            output_plot = gr.Plot()
        
        with gr.TabItem("Description"):
            readme_content = gr.Markdown(load_readme())
    
    predict_button.click(
        predict,
        inputs=input_image,
        outputs=[output_text, output_plot]
    )

demo.launch()