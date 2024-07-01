import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump, load
import librosa
import numpy as np

# Încărcarea datelor
data = pd.read_csv('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\audio_features_dangerous.csv')

# Asigură-te că datele sunt încărcate corect
print(data.head())

# Selectarea featurilor relevante
features = [
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
    'MFCC_11', 'MFCC_12', 'MFCC_13', 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4', 'Chroma_5', 'Chroma_6',
    'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10', 'Chroma_11', 'Chroma_12', 'SpectralContrast_1',
    'SpectralContrast_2', 'SpectralContrast_3', 'SpectralContrast_4', 'SpectralContrast_5', 'SpectralContrast_6',
    'SpectralContrast_7', 'ZeroCrossingRate'
]

X = data[features]
y = data['audio_type']  # etichetele tipului de sunet periculos (glass, gunshot, scream)

# Verifică etichetele pentru a te asigura că sunt corecte
print(y.value_counts())

# Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scalarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crearea și antrenarea modelului SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predicțiile pe setul de test
y_pred = svm_model.predict(X_test_scaled)

# Evaluarea modelului
print("SVM - Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Matrice de confuzie:\n", confusion_matrix(y_test, y_pred))

# Asigură-te că directorul pentru salvarea modelului există
model_dir = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'svm_model_level2.joblib')

# Salvarea modelului SVM
dump(svm_model, model_path)

# Funcția pentru extragerea caracteristicilor dintr-un fișier audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=33)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    features = np.concatenate((
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.mean(zcr, axis=1)
    ))
    return features

# Funcția pentru a prezice tipul sunetului
def predict_sound(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])  # Scalează caracteristicile
    prediction = svm_model.predict(features)
    return prediction[0]

# Funcția principală pentru a testa fișierul audio
def main():
    sound_file_path = input("Enter the path to the sound file: ")
    if os.path.exists(sound_file_path):
        result = predict_sound(sound_file_path)
        print(f"The sound is classified as: {result}")
    else:
        print("The provided file path does not exist. Please check the path and try again.")

if __name__ == "__main__":
    main()
