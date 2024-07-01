import os
import librosa
import noisereduce as nr
from sklearn.preprocessing import MaxAbsScaler
import soundfile as sf

def preprocess_audio_files(audio_folder, export_path):
    """
    Procesează fișierele audio dintr-un director, aplică reducerea zgomotului și normalizarea,
    și salvează rezultatele într-un alt director specificat.
    """
    # Asigurăm că directorul de export există
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Parcurgem fiecare fișier audio în directorul specificat
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            file_path = os.path.join(audio_folder, filename)
            audio, sr = librosa.load(file_path, sr=None)

            # Reducerea zgomotului de fond
            audio_nr = nr.reduce_noise(y=audio, sr=sr)

            # Normalizarea amplitudinii la [-1, 1]
            scaler = MaxAbsScaler()
            audio_scaled = scaler.fit_transform(audio_nr.reshape(-1, 1)).flatten()

            # Extragem numele original al fișierului fără extensie
            original_name = os.path.splitext(filename)[0]

            # Construim numele fișierului pentru fiecare înregistrare preprocesată
            output_file = os.path.join(export_path, f'{original_name}_preprocessed.wav')

            # Scriem fișierul WAV
            sf.write(output_file, audio_scaled, sr)
            print(f"Processed and saved: {output_file}")

# Locații pentru folderul de intrare și de ieșire
audio_folder = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data'
export_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\All_preprocessed_v2'

# Apelarea funcției
preprocess_audio_files(audio_folder, export_path)
