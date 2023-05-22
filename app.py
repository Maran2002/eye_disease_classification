import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from flask import Flask, render_template, request
from models import EyeDiseaseClassification


# Load the trained model and define the transformation pipeline
model_path = 'eye_disease_classification.pkl'
model = torch.load(model_path)
transformations = Compose([
    Resize((32, 32)),
    ToTensor(),
    Normalize((0.5667, 0.5198, 0.4955), (0.229, 0.224, 0.225))
])

# Create the Flask app
app = Flask(__name__)

# Define the route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image from the request
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            # Load and preprocess the input image
            input_image = Image.open(uploaded_file).convert('RGB')
            input_tensor = transformations(input_image)
            input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

            # Make predictions with the model
            output = model(input_batch)
            predicted_class = torch.argmax(output, dim=1).item()

            # Map the predicted class to the corresponding eye disease type
            class_mapping_values = {0: "cataract", 1: "diabetic_retinopathy", 2: "glaucoma", 3: "normal"}
            predicted_disease = class_mapping_values[predicted_class]

            # Render the result page with the predicted disease
            return render_template('result.html', predicted_disease=predicted_disease)

    # Render the index page with the image upload form
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run()
