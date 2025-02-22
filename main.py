import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Set Custom Page Style with Light Blue Background
st.markdown(
    """
    <style>
        /* Apply Light Blue Background */
        .stApp {
            background-color: #add8e6; /* Light Blue */
        }
        
        .main {
            background: url('home.jpg') no-repeat center center fixed;
            background-size: cover;
            padding: 20px;
            border-radius: 10px;
        }
        
        .title {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<p class="title">Fruits And Vegetables Recognition System</p>', unsafe_allow_html=True)
# File Uploader
test_image = st.file_uploader("Choose an Image:")
if test_image:
    st.image(test_image, use_column_width=True)
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        with open("labels.txt") as f:
            label = [i.strip() for i in f.readlines()]
        
        st.success("Model is Predicting it's a {}".format(label[result_index]))

st.markdown('</div>', unsafe_allow_html=True)
