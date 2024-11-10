from flask import Flask, render_template, request, send_file
from keras.models import load_model
from skimage import img_as_ubyte, io, transform
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Load the model (you might want to load this only once, not on every request)
model = load_model(r"D:\DataSets\Tumor\new\PRELU.h5")

def predict_hr_image(lr_image):
    # Preprocess the LR image
    lr_image_resized = transform.resize(lr_image, (64, 64, 1))
    lr_image_resized = np.expand_dims(lr_image_resized, axis=0)

    # Predict using the model
    predicted_hr_image = model.predict(lr_image_resized)
    predicted_hr_image = predicted_hr_image.reshape((256, 256))

    predicted_hr_image = img_as_ubyte((predicted_hr_image - predicted_hr_image.min()) / (predicted_hr_image.max() - predicted_hr_image.min()))

    return predicted_hr_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    img = io.imread(file).astype(np.float32)
    predicted_hr_image = predict_hr_image(img)

    # Convert NumPy array to bytes
    img_bytes = predicted_hr_image.tobytes()

    return send_file(BytesIO(img_bytes), mimetype='image/png', as_attachment=True, attachment_filename="enhanced_image.png")

if __name__ == '__main__':
    app.run(debug=True)
