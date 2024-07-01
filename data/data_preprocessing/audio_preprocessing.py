import os
import librosa
import noisereduce as nr
from sklearn.preprocessing import MaxAbsScaler
import soundfile as sf

def preprocess_audio_files(audio_folder):
    # Lista pentru a stoca datele preprocesate
    preprocessed_data = []

    # Parcurgem fiecare fișier audio în directorul specificat
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):  # Corecția condiției
            # Încărcăm fișierul audio
            file_path = os.path.join(audio_folder, filename)
            audio, sr = librosa.load(file_path, sr=None)

            # Reducerea zgomotului de fond
            audio_nr = nr.reduce_noise(y=audio, sr=sr)

            # Normalizarea amplitudinii la [-1, 1]
            scaler = MaxAbsScaler()
            audio_scaled = scaler.fit_transform(audio_nr.reshape(-1, 1)).flatten()

            # Adăugăm datele preprocesate la listă
            preprocessed_data.append((audio_scaled, sr))

    return preprocessed_data

def save_preprocessed_data(preprocessed_data, export_path):
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    for i, (audio, sr) in enumerate(preprocessed_data):
        # Construim numele fișierului pentru fiecare înregistrare preprocesată
        output_file = os.path.join(export_path, f'preprocessed_{i}.wav')
        # Scriem fișierul WAV
        sf.write(output_file, audio, sr)

# Locații pentru folderul de intrare și de ieșire
audio_folder = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\AudioFiles3'  # Aici introduci calea către fișierele audio
export_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\AudioFiles3\\date_preprocesate'  # Aici introduci calea unde să fie salvate fișierele prelucrate

# Apelarea funcțiilor
preprocessed_data = preprocess_audio_files(audio_folder)
save_preprocessed_data(preprocessed_data, export_path)
