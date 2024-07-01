import os
import joblib
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Directory containing the audio files to be analyzed
recordings_folder = r'C:\Users\Matebook 14s\Desktop\Licenta\recordings\PC_recordings'

# Load the models
model_rf = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\random_forest_model.joblib')
model_rf_layer2 = joblib.load(
    'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\random_forest_model_level2.joblib')
model_svm = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model.joblib')
model_svm_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model_level2.joblib')
model_knn = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\knn_model.joblib')
model_knn_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\knn_model_level2.joblib')
model_nn = load_model('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\audio_classification_model.keras')

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


# Function to extract features
def extract_features_for_prediction(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    if audio.size == 0:
        print(f"Warning: {file_path} is empty or could not be read.")
        return None

    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    features.extend(mfccs_processed)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    features.extend(chroma_processed)

    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=6, fmin=200.0)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    features.extend(spec_contrast_processed)

    # Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features.append(zero_crossing_processed)

    feature_array = np.array(features).reshape(1, -1)
    return feature_array


# Prediction functions for each model
def predict_rf(features):
    features_scaled = scaler.transform(features)
    is_dangerous = model_rf.predict(features_scaled)[0]
    if is_dangerous == 'dangerous':
        sound_type = model_rf_layer2.predict(features_scaled)[0]
        return "Dangerous", sound_type, "Random Forest"
    return "Not Dangerous", "None", "Random Forest"


def predict_svm(features):
    features_scaled = scaler.transform(features)
    is_dangerous = model_svm.predict(features_scaled)[0]
    if is_dangerous == 0:  # Assuming 0 is dangerous, based on previous classification
        sound_type = model_svm_layer2.predict(features_scaled)[0]
        return "Dangerous", sound_type, "SVM"
    return "Not Dangerous", "None", "SVM"


def predict_knn(features):
    features_scaled = scaler.transform(features)
    is_dangerous = model_knn.predict(features_scaled)[0]
    if is_dangerous == 'dangerous':
        sound_type = model_knn_layer2.predict(features_scaled)[0]
        return "Dangerous", sound_type, "KNN"
    return "Not Dangerous", "None", "KNN"


def predict_nn(features):
    features_scaled = scaler.transform(features)
    prediction = model_nn.predict(features_scaled)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label


# Initialize counters for the report
report = {
    "Random Forest": {"Dangerous": 0, "Sound Types": {}},
    "SVM": {"Dangerous": 0, "Sound Types": {}},
    "KNN": {"Dangerous": 0, "Sound Types": {}},
    "Neural Network": {"Dangerous": 0, "Sound Types": {}}
}

# Process each audio file and update the report
for file_name in os.listdir(recordings_folder):
    file_path = os.path.join(recordings_folder, file_name)
    features = extract_features_for_prediction(file_path)

    if features is None:
        continue

    for predict_fn in [predict_rf, predict_svm, predict_knn]:
        result, sound_type, model_name = predict_fn(features)
        if result == "Dangerous":
            report[model_name]["Dangerous"] += 1
            if sound_type in report[model_name]["Sound Types"]:
                report[model_name]["Sound Types"][sound_type] += 1
            else:
                report[model_name]["Sound Types"][sound_type] = 1

    predicted_label = predict_nn(features)
    if predicted_label is not None:
        decoded_label = label_encoder_combined.inverse_transform(predicted_label)[0]
        label_part = decoded_label // 10
        audio_type_part = decoded_label % 10
        label = label_encoder_label.inverse_transform([label_part])[0]
        audio_type = label_encoder_audio_type.inverse_transform([audio_type_part])[0]
        if label == "dangerous":
            report["Neural Network"]["Dangerous"] += 1
            if audio_type in report["Neural Network"]["Sound Types"]:
                report["Neural Network"]["Sound Types"][audio_type] += 1
            else:
                report["Neural Network"]["Sound Types"][audio_type] = 1

# Print the report
for model_name, data in report.items():
    print(f"Model: {model_name}")
    print(f"  Dangerous predictions: {data['Dangerous']}")
    print("  Sound Types:")
    for sound_type, count in data["Sound Types"].items():
        print(f"    {sound_type}: {count}")
    print()
