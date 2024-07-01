import joblib
import librosa
import numpy as np
import os
import smtplib
import pandas as pd
import ssl
from email.message import EmailMessage
from config.app_config import Email_App_Password, Email_Username, Email_Recievers
# Load the models
model_rf = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\random_forest_model.joblib')
model_rf_layer2 = joblib.load(
    'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\random_forest_model_level2.joblib')
model_svm = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model.joblib')
model_svm_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model_level2.joblib')
model_knn = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\knn_model.joblib')
model_knn_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\knn_model_level2.joblib')


def extract_features_for_prediction(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
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
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    features.extend(spec_contrast_processed)

    # Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features.append(zero_crossing_processed)

    feature_names = [
        'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6',
        'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13',
        'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4', 'Chroma_5',
        'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10',
        'Chroma_11', 'Chroma_12',
        'SpectralContrast_1', 'SpectralContrast_2', 'SpectralContrast_3',
        'SpectralContrast_4', 'SpectralContrast_5', 'SpectralContrast_6',
        'SpectralContrast_7',
        'ZeroCrossingRate'
    ]

    feature_array = np.array(features).reshape(1, -1)
    return pd.DataFrame(feature_array, columns=feature_names)

def predict_rf(file_path):
    """ Predict if a sound is dangerous and classify the type if it is. """
    features = extract_features_for_prediction(file_path)
    is_dangerous = model_rf.predict(features)[0]

    if is_dangerous == 'dangerous':
        sound_type = model_rf_layer2.predict(features)[0]
        return "Dangerous", sound_type, "Random Forest"
    return "Not Dangerous", "None", "Random Forest"

def predict_svm(file_path):
    """ Predict using the trained SVM model. """
    features = extract_features_for_prediction(file_path)
    is_dangerous = model_svm.predict(features)[0]

    if is_dangerous == 'dangerous':
        sound_type = model_svm_layer2.predict(features)[0]
        return "Dangerous", sound_type, "SVM"
    return "Not Dangerous", "None", "SVM"

def predict_knn(file_path):
    """ Predict using the trained KNN model. """
    features = extract_features_for_prediction(file_path)
    is_dangerous = model_knn.predict(features)[0]

    if is_dangerous == 'dangerous':
        sound_type = model_knn_layer2.predict(features)[0]
        return "Dangerous", sound_type, "KNN"
    return "Not Dangerous", "None", "KNN"

def send_email(to_email, subject, body):
    """ Send an email with the given subject and body. """
    em = EmailMessage()
    em['From'] = Email_Username
    em['To'] = to_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(Email_Username, Email_App_Password)
        smtp.send_message(em)

def analyze_and_send_results(file_path):
    # Initialize results
    results = []

    # Predict using Random Forest
    result_rf, type_rf, model_rf_name = predict_rf(file_path)
    results.append((result_rf, type_rf, model_rf_name))

    # Predict using SVM
    result_svm, type_svm, model_svm_name = predict_svm(file_path)
    results.append((result_svm, type_svm, model_svm_name))

    # Predict using KNN
    result_knn, type_knn, model_knn_name = predict_knn(file_path)
    results.append((result_knn, type_knn, model_knn_name))

    # Send first email: whether the sound is dangerous or not
    body_first_email = "Results of sound analysis:\n\n"
    for result, sound_type, model_name in results:
        body_first_email += f"Model: {model_name}\nResult: {result}\n\n"
    send_email(Email_Recievers, "Sound Danger Classification", body_first_email)

    # Check if any model classified the sound as dangerous
    dangerous_classifications = [(result, sound_type, model_name) for result, sound_type, model_name in results if result == "Dangerous"]

    # Send second email only if there are dangerous classifications
    if dangerous_classifications:
        body_second_email = "Detailed sound classification:\n\n"
        for result, sound_type, model_name in dangerous_classifications:
            body_second_email += f"Model: {model_name}\nSound Type: {sound_type}\n\n"
        send_email(Email_Recievers, "Detailed Sound Classification", body_second_email)


sound_file_path = r'C:\Users\Matebook 14s\Desktop\Licenta\data_test\All_preprocessed_v2\danger_scream_87_preprocessed.wav'
analyze_and_send_results(sound_file_path)
