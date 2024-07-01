import librosa
import numpy as np
import pandas as pd
import os

from config.app_config import MP3

def extract_features(file_path):
    """
    Extrage caracteristici audio dintr-un fișier dat și returnează un dictionar cu valori.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    features = {}

    # Extragem MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    for i, mfcc in enumerate(mfccs_processed):
        features[f'MFCC_{i + 1}'] = mfcc

    # Extragem chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    for i, chrom in enumerate(chroma_processed):
        features[f'Chroma_{i + 1}'] = chrom

    # Extragem spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    for i, contrast in enumerate(spec_contrast_processed):
        features[f'SpectralContrast_{i + 1}'] = contrast

    # Extragem zero crossing rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features['ZeroCrossingRate'] = zero_crossing_processed

    return features


def process_directory(directory):
    """
    Procesează un director întreg, extrage caracteristicile pentru fiecare
    fișier audio și le salvează într-un DataFrame.
    """
    data = []
    for file in os.listdir(directory):
        if file.endswith((MP3, '.wav')):
            file_path = os.path.join(directory, file)
            try:
                features = extract_features(file_path)
                features['label'] = get_label(file)
                data.append(features)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(data)
    return df


def get_label(file_name):
    """
    Determină eticheta din numele fișierului.
    """
    if 'danger' in file_name:
        return 'dangerous'
    elif 'safe' in file_name:
        return 'safe'
    return 'unknown'


def save_features_to_csv(df, csv_filename, save_path="."):
    """
    Salvează caracteristicile extrase într-un fișier CSV la o cale specificată.
    """
    full_path = os.path.join(save_path, csv_filename)
    df.to_csv(full_path, index=False)
    print(f"Features saved to {full_path}")


# Utilizare
audio_directory = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\All_Preprocesed'
output_csv = 'audio_features.csv'
save_directory = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test'
df = process_directory(audio_directory)
save_features_to_csv(df, output_csv, save_directory)
