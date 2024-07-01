import os
import re


def rename_files_with_pattern(audio_folder, pattern, prefix, start_number=1):
    counter = start_number
    regex = re.compile(pattern, re.IGNORECASE)
    renamed_files = set()

    # Colectează toate fișierele care trebuie procesate
    files = [f for f in os.listdir(audio_folder) if f.endswith(('.mp3', '.wav'))]

    # Procesează și redenumește fișierele conform pattern-ului dat
    for filename in files:
        if regex.search(filename):
            new_filename = f"{prefix}_{str(counter).zfill(2)}{os.path.splitext(filename)[1]}"
            old_file = os.path.join(audio_folder, filename)
            new_file = os.path.join(audio_folder, new_filename)

            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")
            counter += 1
            renamed_files.add(new_filename)  # Adaugă la setul de fișiere redenumite

    return renamed_files


def rename_remaining_safe(audio_folder, renamed_files, start_number=1):
    counter = start_number
    files = [f for f in os.listdir(audio_folder) if f.endswith(('.mp3', '.wav'))]

    for filename in files:
        if filename not in renamed_files:
            new_filename = f"safe_{str(counter).zfill(2)}{os.path.splitext(filename)[1]}"
            old_file = os.path.join(audio_folder, filename)
            new_file = os.path.join(audio_folder, new_filename)

            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")
            counter += 1


# Setează folderul și apelează funcțiile
audio_folder = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data'

renamed_files = set()
renamed_files.update(rename_files_with_pattern(audio_folder, r"(gun|shot|gunshot)", "danger_gunshot"))
renamed_files.update(rename_files_with_pattern(audio_folder, r"glass", "danger_glass", start_number=1))
renamed_files.update(rename_files_with_pattern(audio_folder, r"(scream|screaming)", "danger_scream", start_number=1))

# Redenumirea fișierelor rămase ca "safe"
rename_remaining_safe(audio_folder, renamed_files, start_number=1)
