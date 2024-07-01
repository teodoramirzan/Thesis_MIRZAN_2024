import requests
import os

# Tokenul de acces OAuth obținut după autentificare
access_token = 'DUxqOLh1HFZPNPZDzloP8iW2IIyg5o' #token

# Query-ul dorit
query = 'kids playing'
num_sounds_to_download = 330 # numărul total de sunete pe care vrei să le descarci
file_extension = 'wav'

# Folderul unde să salvezi sunetele
save_folder = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\AudioFiles3'

# Verifică dacă folderul există, altfel îl creează
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# URL-ul inițial pentru căutarea sunetelor pe Freesound
base_search_url = 'https://freesound.org/apiv2/search/text/'
params = {
    'query': query,
    'filter': f'type:{file_extension}',
    'fields': 'id,name,previews',
    'page_size': num_sounds_to_download  # Numărul maxim de sunete per pagină
}

# Header-ul pentru autorizare
headers = {
    'Authorization': f'Bearer {access_token}'
}

# Face cererea de căutare
def fetch_sounds(search_url, params, headers):
    sounds_downloaded = 0
    while sounds_downloaded < num_sounds_to_download:
        response = requests.get(search_url, params=params, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            for sound in response_data['results']:
                if sounds_downloaded >= num_sounds_to_download:
                    break
                # URL-ul pentru preview-ul sunetului
                preview_url = sound['previews']['preview-hq-mp3']
                # Descarcă sunetul
                r = requests.get(preview_url, stream=True)
                if r.status_code == 200:
                    file_path = os.path.join(save_folder, f"{sound['id']}_{sound['name']}.mp3")
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Descărcat: {file_path}")
                    sounds_downloaded += 1
                else:
                    print(f"Eroare la descărcarea sunetului: {sound['name']}")
            # Verifică dacă există o pagină următoare
            if 'next' in response_data and sounds_downloaded < num_sounds_to_download:
                search_url = response_data['next']
                params = None  # Parametrii nu mai sunt necesari pentru URL-ul completat
            else:
                break
        else:
            print("Eroare la cererea de căutare:", response.status_code, response.text)
            break

# Începe procesul de descărcare
fetch_sounds(base_search_url, params, headers)

import requests
import os
from pydub import AudioSegment

# Tokenul de acces OAuth obținut după autentificare
access_token = 'YOUR_ACCESS_TOKEN'

# Query-ul dorit și numărul de sunete de descărcat
query = 'QUERY_TERM'
num_sounds_to_download = 5
original_extension = 'flac'  # Extensia originală pe care o căutăm

# Folderul unde să salvezi sunetele
save_folder = 'path_to_save_sounds'

# Verifică dacă folderul există, altfel îl creează
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# URL-ul pentru căutarea sunetelor pe Freesound
search_url = f'https://freesound.org/apiv2/search/text/?query={query}&filter=type:{original_extension}&fields=id,name,previews'

# Header-ul pentru autorizare
headers = {
    'Authorization': f'Bearer {access_token}'
}

# Face cererea de căutare
response = requests.get(search_url, headers=headers)

if response.status_code == 200:
    # Procesează răspunsul primit și obține sunetele
    sounds = response.json()['results'][:num_sounds_to_download]

    for sound in sounds:
        # URL-ul pentru descărcarea sunetului
        sound_url = sound['previews']['preview-hq-mp3']

        # Descarcă sunetul
        r = requests.get(sound_url, stream=True)
        if r.status_code == 200:
            original_file_path = os.path.join(save_folder, f"{sound['id']}_{sound['name']}.{original_extension}")
            with open(original_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Convertirea sunetului în .wav folosind pydub
            wav_file_path = original_file_path.replace(f".{original_extension}", ".wav")
            AudioSegment.from_file(original_file_path).export(wav_file_path, format='wav')

            print(f"Descărcat și convertit: {wav_file_path}")

            # Opțional: șterge fișierul original dacă nu este necesar
            os.remove(original_file_path)
        else:
            print(f"Eroare la descărcarea sunetului: {sound['name']}")
else:
    print("Eroare la cererea de căutare:", response.status_code, response.text)
