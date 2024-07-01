# audio_recorder.py
import pyaudio
import wave
import os

def start_recording(duration=5, sample_rate=44100, chunk_size=1024):
    """
    Înregistrează audio de la microfonul dispozitivului pentru o durată specificată și
    salvează înregistrarea într-un fișier WAV temporar.

    Args:
        duration (int): Durata înregistrării în secunde.
        sample_rate (int): Rata de eșantionare a înregistrării.
        chunk_size (int): Numărul de cadre audio procesate într-un singur apel.

    Returns:
        str: Calea către fișierul audio înregistrat.
    """
    audio_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2  # Stereo

    p = pyaudio.PyAudio()

    # Deschide un stream pentru înregistrare
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Începe înregistrarea audio...")
    frames = []

    # Capturarea datelor audio
    for i in range(int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    # Închide stream-ul și termină PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Înregistrare completă.")

    # Salvarea înregistrării într-un fișier WAV
    save_dir = r'C:\Users\Matebook 14s\Desktop\Licenta\real_time_sounds'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'recorded.wav')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename
