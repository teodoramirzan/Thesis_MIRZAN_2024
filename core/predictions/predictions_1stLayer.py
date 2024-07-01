import joblib
import librosa
import numpy as np
import os
import smtplib
import ssl
from email.message import EmailMessage
from config.app_config import Email_App_Password, Email_Username, Email_Recievers

# Load the RandomForest model
model_rf = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model.joblib')


def extract_features_for_prediction(file_path):
    """ Extract features from an audio file for prediction, matching the training feature extraction. """
    audio, sample_rate = librosa.load(file_path, sr=None)
    features = []

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    features.extend(mfccs_processed)

    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    features.extend(chroma_processed)

    # Extract Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    features.extend(spec_contrast_processed)

    # Extract Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features.append(zero_crossing_processed)

    return np.array(features).reshape(1, -1)  # Reshape for single prediction


def predict_sound(file_path):
    """ Predict if a sound is dangerous using the loaded model and send an email if it is. """
    features = extract_features_for_prediction(file_path)
    is_dangerous = model_rf.predict(features)[0]

    if is_dangerous == 'dangerous':
        send_email('Safety Alert', 'This sound has been classified as dangerous.')


def send_email(subject, body):
    """ Send an email with the given subject and body. """
    em = EmailMessage()
    em['From'] = Email_Username
    em['To'] = Email_Recievers
    em['Subject'] = subject
    em.set_content(body)

    # Add SSL (layer of security)
    context = ssl.create_default_context()

    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(Email_Username, Email_App_Password)
        smtp.send_message(em)


# Example usage
sound_file_path = 'C:\\Users\\Matebook 14s\\Downloads\\734097__modusmogulus__gunshot-forest-2m-harsh.wav'
predict_sound(sound_file_path)

