import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import librosa

# Define the path to the saved model
model_save_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\audio_classification_model.keras'

# Load the trained model
model = load_model(model_save_path)

# Function to extract audio features
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)

        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(zero_crossing_rate)
        ])
        return features
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(e)
        return None

# Function to predict the class of an audio file
def predict_audio(file_name, model, scaler):
    features = extract_features(file_name)
    if features is not None:
        features = scaler.transform([features])
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)
        return predicted_label
    else:
        return None

# Define the audio file path
audio_file_path = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\recordings\\PC_recordings\\tst.wav"

# Load the scaler used during training
scaler_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoders used during training
label_encoder_combined_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\label_encoder_combined.pkl'
label_encoder_label_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\label_encoder_label.pkl'
label_encoder_audio_type_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\label_encoder_audio_type.pkl'
with open(label_encoder_combined_path, 'rb') as f:
    label_encoder_combined = pickle.load(f)
with open(label_encoder_label_path, 'rb') as f:
    label_encoder_label = pickle.load(f)
with open(label_encoder_audio_type_path, 'rb') as f:
    label_encoder_audio_type = pickle.load(f)

# Predict the class of the audio file
predicted_label = predict_audio(audio_file_path, model, scaler)
if predicted_label is not None:
    decoded_label = label_encoder_combined.inverse_transform(predicted_label)[0]
    label_part = decoded_label // 10
    audio_type_part = decoded_label % 10
    label = label_encoder_label.inverse_transform([label_part])[0]
    audio_type = label_encoder_audio_type.inverse_transform([audio_type_part])[0]
    print(f"Predicted label for the audio file: {label}, Audio type: {audio_type}")
else:
    print("Failed to predict the audio file.")
