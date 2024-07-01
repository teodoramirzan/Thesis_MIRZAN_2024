# main.py
from data.data_preprocessing.audio_preprocessing import preprocess_audio_files, save_preprocessed_data

def main():
    # Calea către directorul cu fișiere audio
    audio_folder = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\AudioFiles3'
    # Calea unde vrei să salvezi fișierele preprocesate
    export_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\AudioFiles3\\date_preprocesate'

    # Preprocesăm fișierele audio
    preprocessed_data = preprocess_audio_files(audio_folder)
    # Salvăm fișierele audio preprocesate
    save_preprocessed_data(preprocessed_data, export_path)

if __name__ == '__main__':
    main()
