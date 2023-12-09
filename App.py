import streamlit as st
from keras.models import load_model
from skimage import img_as_ubyte, io, transform
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def predict_hr_image(model, lr_image):
    # Preprocess the LR image
    lr_image_resized = transform.resize(lr_image, (64, 64, 1))
    lr_image_resized = np.expand_dims(lr_image_resized, axis=0)

    # Predict using the model
    predicted_hr_image = model.predict(lr_image_resized)
    predicted_hr_image = predicted_hr_image.reshape((256, 256))
    
    predicted_hr_image = img_as_ubyte((predicted_hr_image - predicted_hr_image.min()) / (predicted_hr_image.max() - predicted_hr_image.min()))

    return predicted_hr_image

st.set_page_config(page_title="MRI Image Super Resolution", layout="wide")

st.title("MRI Image Super Resolution")

input_image = st.file_uploader(label='Upload Low resolution image', type=["jpg", "png", "jpeg", 'webp'])

# Load the model (you might want to load this only once, not on every app rerun)
model = load_model(r"D:\DataSets\Tumor\new\PRELU.h5")

col1, col2 = st.columns([10, 10])

with col1:
    if input_image:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(io.imread(input_image), cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
        
        if st.button("Enhance"):
            # Read and preprocess the uploaded image
            img = io.imread(input_image).astype(np.float32)

            # Predict using the model
            predicted_hr_image = predict_hr_image(model, img)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(predicted_hr_image, cmap="gray")
                ax.axis("off")
                st.pyplot(fig)

                # Convert NumPy array to bytes
                img_bytes = predicted_hr_image.tobytes()

                # Add a download button for the enhanced image
                download_btn = st.download_button(
                    label="Download Enhanced Image",
                    data=BytesIO(img_bytes),
                    file_name="enhanced_image.png",
                    mime="image/png"
                )
