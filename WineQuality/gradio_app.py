import gradio as gr
import pickle
import numpy as np

# Load your trained model and label encoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the prediction function
def predict_wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                         density, pH, sulphates, alcohol):
    # Prepare input data for prediction
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                          density, pH, sulphates, alcohol]])
    
    # Make prediction
    prediction_encoded = model.predict(features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    
    return prediction_label

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_wine_quality,
    inputs=[
        gr.Number(label="Fixed Acidity"),
        gr.Number(label="Volatile Acidity"),
        gr.Number(label="Citric Acid"),
        gr.Number(label="Residual Sugar"),
        gr.Number(label="Chlorides"),
        gr.Number(label="Free Sulfur Dioxide"),
        gr.Number(label="Total Sulfur Dioxide"),
        gr.Number(label="Density"),
        gr.Number(label="pH"),
        gr.Number(label="Sulphates"),
        gr.Number(label="Alcohol")
    ],
    outputs="text",
    title="Wine Quality Prediction",
    description="Enter the wine's chemical properties to predict its quality."
)

# Launch the Gradio app with sharing enabled
interface.launch(share=True)
