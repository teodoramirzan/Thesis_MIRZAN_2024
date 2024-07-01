import os
import warnings
import numpy as np
import librosa
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Definim categoriile și etichetăm datele
categories = {'gunshot': 0, 'glass': 1, 'scream': 2, 'safe': 3}
data_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\All_preprocessed_v2'

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma = np.mean(chroma.T, axis=0)
    # Extract Spectral Contrast
    # Adjust parameters to avoid exceeding the Nyquist frequency
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, fmin=20.0, n_bands=6)
    spec_contrast = np.mean(spec_contrast.T, axis=0)
    # Extract Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr = np.mean(zcr)

    return np.concatenate([mfccs, chroma, spec_contrast, [zcr]])

def load_data(data_path):
    features = []
    labels = []
    for file_name in os.listdir(data_path):
        if file_name.endswith(".wav"):
            parts = file_name.split('_')
            category = parts[1] if parts[0] == 'danger' else 'safe'
            if category in categories:
                file_path = os.path.join(data_path, file_name)
                feature_vector = extract_features(file_path)
                features.append(feature_vector)
                labels.append(categories[category])
    return np.array(features), np.array(labels)

# Ignore warnings about empty frequency sets
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# Încărcăm datele și le împărțim în seturi de antrenament și testare
X, y = load_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertim etichetele în categorii (one-hot encoding)
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

# Construim modelul
model = Sequential([
    Dense(256, input_shape=(33,), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

# Compilăm modelul
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Antrenăm modelul
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluăm modelul
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
