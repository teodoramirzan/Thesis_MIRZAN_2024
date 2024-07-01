import joblib
import librosa
import numpy as np
import os

# Încarcă modelul SVM antrenat
model_path = r"C:\Users\Matebook 14s\Desktop\Licenta\saved_models\old\svm_model.joblib"
svm_model = joblib.load(model_path)

# Funcție pentru extragerea caracteristicilor dintr-un fișier audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=33)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Funcție pentru a prezice dacă sunetul este periculos sau nu
def predict_sound(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)  # Reshape pentru model
    prediction = svm_model.predict(features)
    return "Dangerous" if prediction[0] == 0 else "Not Dangerous"

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
