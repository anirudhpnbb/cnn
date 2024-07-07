from flask import Flask, request, jsonify
from PIL import Image
import torch

from data_preprocessing import transform
from detection import PneumoniaDetectionCNN

app = Flask(__name__)
model = PneumoniaDetectionCNN()
model.load_state_dict(torch.load)("pneumonia_detection_model.pth")
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return jsonify({'prediction': 'Pneumonia' if pred.item() == 1 else 'Normal'})

if __name__ == '__main__':
    app.run(debug=True)