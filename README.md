# fruit-and-vegetable-recognition-system

This project is a machine learning-based application that recognizes different fruits and vegetables from images using a trained deep learning model.

## Features

- Image classification using a Convolutional Neural Network (CNN)
- Web interface built with Streamlit for user interaction
- Supports uploading images for prediction
- Pre-trained model (`trained_model.h5`) for quick inference

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- NumPy
- Streamlit

### Clone the Repository

```bash
git clone https://github.com/naveenreddy03/fruit-and-vegetable-recognition-system
cd FruitsVegRecognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application

```bash
streamlit run main.py
```

### Upload an Image

- Open the Streamlit web app
- Upload an image of a fruit or vegetable
- Click "Predict" to get the classification result

## Project Structure

```
├── Training_fruit_veg.ipynb  # Jupyter Notebook for training the model
├── trained_model.h5          # Pre-trained model
├── main.py                   # Streamlit application
├── labels.txt                # Labels for classification
├── requirements.txt          # Required dependencies
├── README.md                 # Project documentation
```

## Model Details

The model was trained using a Convolutional Neural Network (CNN) architecture.
The dataset includes various fruits and vegetables, and the model was optimized for accurate classification.

## License

This project is licensed under the MIT License - see the LICENSE file for details

