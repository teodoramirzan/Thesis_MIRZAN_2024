import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from config.app_config import MP3

def extract_features(file_path):
    """
    Extracts audio features from a given file and returns a dictionary of values.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    features = {}

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    for i, mfcc in enumerate(mfccs_processed):
        features[f'MFCC_{i + 1}'] = mfcc

    # Extract chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    for i, chrom in enumerate(chroma_processed):
        features[f'Chroma_{i + 1}'] = chrom

    # Extract spectral contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    for i, contrast in enumerate(spec_contrast_processed):
        features[f'SpectralContrast_{i + 1}'] = contrast

    # Extract zero crossing rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features['ZeroCrossingRate'] = zero_crossing_processed

    return features

def get_label(file_name):
    """
    Determines the label from the filename.
    """
    if 'danger' in file_name:
        return 'dangerous'
    elif 'safe' in file_name:
        return 'safe'
    return 'unknown'

def get_audio_type(file_name):
    """
    Extracts the type of dangerous audio from the file name.
    """
    parts = file_name.split('_')
    if 'danger' in parts[0]:
        return parts[1]  # Assumes the type is the second element after 'danger'
    return 'none'

def process_directory(directory):
    """
    Processes an entire directory, extracts features for each audio file, and saves them to a DataFrame.
    """
    data = []
    for file in os.listdir(directory):
        if file.endswith((MP3, '.wav')):
            file_path = os.path.join(directory, file)
            try:
                features = extract_features(file_path)
                features['label'] = get_label(file)
                features['audio_type'] = get_audio_type(file)
                data.append(features)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(data)
    return df

def save_features_to_csv(df, csv_filename, save_path="."):
    """
    Saves the extracted features to a CSV file at a specified path.
    """
    full_path = os.path.join(save_path, csv_filename)
    df.to_csv(full_path, index=False)
    print(f"Features saved to {full_path}")

# Usage
audio_directory = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\Dangerous\\Dangerous_Preprocessed'
output_csv = 'audio_features_dangerous.csv'
save_directory = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test'
df = process_directory(audio_directory)
save_features_to_csv(df, output_csv, save_directory)

# Load the extracted features for model training and evaluation
data = pd.read_csv(os.path.join(save_directory, output_csv))

# Extract features and labels
X = data.drop(columns=['label'])
y = data['label']

# Convert labels to numeric format
y = y.astype('category').cat.codes

# One-hot encoding for categorical features
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv_scores = cross_val_score(model_rf, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# Train the model on the full training set
model_rf.fit(X_train, y_train)

# Save the trained model
model_path = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model_1st.joblib"
joblib.dump(model_rf, model_path)

print("Modelul Random Forest pentru nivelul 1 a fost antrenat și salvat la:", model_path)

# Make predictions on the test set
y_pred = model_rf.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=['not dangerous', 'dangerous'])
print("\nRaport de clasificare:\n", report)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de confuzie:\n", conf_matrix)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAcuratețea modelului: {:.2f}%".format(accuracy * 100))
