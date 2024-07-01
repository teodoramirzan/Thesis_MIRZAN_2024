import os
import librosa
import noisereduce as nr
from sklearn.preprocessing import MaxAbsScaler
import soundfile as sf
import numpy as np
import joblib
from email.message import EmailMessage
import smtplib
import ssl

from config.app_config import Email_Username, Email_App_Password, Email_Recievers
from core.classification_first.predictions_2ndLayer import send_email

# Cărți model și alte resurse
model_rf = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model.joblib')
model_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model.joblib')

# Configurație pentru email (adaptată după nevoile tale)
email_sender = Email_Username
email_password = Email_App_Password
email_receiver = Email_Recievers


def preprocess_and_extract_features(audio_data, sample_rate):
    # Reducerea zgomotului
    audio_nr = nr.reduce_noise(y=audio_data, sr=sample_rate)

    # Normalizare
    scaler = MaxAbsScaler()
    audio_scaled = scaler.fit_transform(audio_nr.reshape(-1, 1)).flatten()

    # Extracția de caracteristici
    mfccs = librosa.feature.mfcc(y=audio_scaled, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_scaled, sr=sample_rate)
    spec_contrast = librosa.feature.spectral_contrast(y=audio_scaled, sr=sample_rate)
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio_scaled)

    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spec_contrast, axis=1),
        np.mean(zero_crossing)
    ])

    return features.reshape(1, -1)


def predict(features):
    is_dangerous = model_rf.predict(features)[0]
    sound_type = "Not Dangerous"
    if is_dangerous == 'dangerous':
        sound_type = model_layer2.predict(features)[0]
    return is_dangerous, sound_type




def process_and_predict(sound_file):
    # Încărcare fișier
    audio_data, sample_rate = librosa.load(sound_file, sr=None)
    features = preprocess_and_extract_features(audio_data, sample_rate)

    # Predicție
    is_dangerous, sound_type = predict(features)

    # Trimite email dacă sunetul este periculos
    if is_dangerous == 'dangerous':
        send_email('Safety Alert', 'A dangerous sound has been detected.')
        # Classify the type of dangerous sound
        sound_type = model_layer2.predict(features)[0]
        send_email('Dangerous Sound Detected', f'The detected sound has been classified as: {sound_type}')

    return is_dangerous, sound_type
