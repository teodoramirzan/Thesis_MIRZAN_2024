import os
import joblib
import librosa
import numpy as np
import pandas as pd
import smtplib
import ssl
import wave
import pyaudio
import paramiko
import pickle
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from email.message import EmailMessage
from config.app_config import Email_App_Password, Email_Username

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RECORDINGS_FOLDER'] = r'C:/Users/Matebook 14s/Desktop/Licenta/recordings/PC_recordings'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RECORDINGS_FOLDER'], exist_ok=True)
app.secret_key = 'your_secret_key'

# Load the models
model_rf = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model_1st.joblib')
model_rf_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model_2nd.joblib')
model_svm = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\svm_model_1st.joblib')
model_svm_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\svm_model_2nd.joblib')
model_knn = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\knn_model_1st.joblib')
model_knn_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\knn_model_2nd.joblib')

# Manually specify the feature columns
feature_columns = [
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6',
    'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13',
    'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4', 'Chroma_5',
    'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10',
    'Chroma_11', 'Chroma_12',
    'SpectralContrast_1', 'SpectralContrast_2', 'SpectralContrast_3',
    'SpectralContrast_4', 'SpectralContrast_5', 'SpectralContrast_6',
    'SpectralContrast_7',
    'ZeroCrossingRate'
]

def extract_features_for_prediction(file_path):
    print(f"Extracting features from file: {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=None)
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    features.extend(mfccs_processed)
    print(f"MFCCs: {mfccs_processed}")

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    features.extend(chroma_processed)
    print(f"Chroma: {chroma_processed}")

    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, fmin=200.0, n_bands=6)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    features.extend(spec_contrast_processed)
    print(f"Spectral Contrast: {spec_contrast_processed}")

    # Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features.append(zero_crossing_processed)
    print(f"Zero Crossing Rate: {zero_crossing_processed}")

    feature_array = np.array(features).reshape(1, -1)
    print(f"Feature Array: {feature_array}")
    return pd.DataFrame(feature_array, columns=feature_columns)

def predict_rf(file_path):
    print(f"Predicting using Random Forest for file: {file_path}")
    features = extract_features_for_prediction(file_path)
    print(f"Features for Random Forest: {features}")
    is_dangerous = model_rf.predict(features)[0]
    print(f"Random Forest Prediction: {is_dangerous}")

    if is_dangerous == 1:
        sound_type = model_rf_layer2.predict(features)[0]
        print(f"Random Forest Layer 2 Prediction: {sound_type}")
        return "Dangerous", sound_type, "Random Forest"
    return "Not Dangerous", "None", "Random Forest"

def predict_svm(file_path):
    print(f"Predicting using SVM for file: {file_path}")
    features = extract_features_for_prediction(file_path)
    print(f"Features for SVM: {features}")
    is_dangerous = model_svm.predict(features)[0]
    print(f"SVM Prediction: {is_dangerous}")

    if is_dangerous == 1:
        sound_type = model_svm_layer2.predict(features)[0]
        print(f"SVM Layer 2 Prediction: {sound_type}")
        return "Dangerous", sound_type, "SVM"
    return "Not Dangerous", "None", "SVM"

def predict_knn(file_path):
    print(f"Predicting using KNN for file: {file_path}")
    features = extract_features_for_prediction(file_path)
    print(f"Features for KNN: {features}")
    is_dangerous = model_knn.predict(features)[0]
    print(f"KNN Prediction: {is_dangerous}")

    if is_dangerous == 1:
        sound_type = model_knn_layer2.predict(features)[0]
        print(f"KNN Layer 2 Prediction: {sound_type}")
        return "Dangerous", sound_type, "KNN"
    return "Not Dangerous", "None", "KNN"

def send_email(to_email, subject, body):
    print(f"Sending email to {to_email} with subject {subject}")
    em = EmailMessage()
    em['From'] = Email_Username
    em['To'] = to_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(Email_Username, Email_App_Password)
        smtp.send_message(em)
    print(f"Email sent to {to_email} with subject: {subject}")

def analyze_and_send_results(file_path, email_receiver):
    print(f"Analyzing and sending results for file: {file_path} to {email_receiver}")
    results = []

    result_rf, type_rf, model_rf_name = predict_rf(file_path)
    results.append((result_rf, type_rf, model_rf_name))

    result_svm, type_svm, model_svm_name = predict_svm(file_path)
    results.append((result_svm, type_svm, model_svm_name))

    result_knn, type_knn, model_knn_name = predict_knn(file_path)
    results.append((result_knn, type_knn, model_knn_name))

    print(f"Results: {results}")

    dangerous_classifications = [(result, sound_type, model_name) for result, sound_type, model_name in results if result == "Dangerous"]
    print(f"Dangerous classifications: {dangerous_classifications}")

    if dangerous_classifications:
        body_first_email = "Results of sound analysis:\n\n"
        for result, sound_type, model_name in results:
            body_first_email += f"Model: {model_name}\nResult: {result}\n\n"
        send_email(email_receiver, "Sound Danger Classification", body_first_email)

        body_second_email = "Detailed sound classification:\n\n"
        for result, sound_type, model_name in dangerous_classifications:
            body_second_email += f"Model: {model_name}\nSound Type: {sound_type}\n\n"
        send_email(email_receiver, "Detailed Sound Classification", body_second_email)

def record_audio(file_name, duration=5):
    audio_format = pyaudio.paInt16
    channels = 2
    sample_rate = 44100
    chunk_size = 1024
    file_path = os.path.join(app.config['RECORDINGS_FOLDER'], file_name)

    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

    print("Recording...")
    frames = []
    for _ in range(int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Recording completed.")

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return file_path

def record_audio_raspberry_pi(file_name, duration=5):
    file_path = f"/home/teo/Documents/{file_name}"
    record_cmd = f"arecord -D hw:1,0 -f cd -t wav -d {duration} -r 44100 {file_path}"
    scp_cmd = f"scp teo@172.20.10.13:{file_path} D:\\{file_name}"
    destination = os.path.join(app.config['RECORDINGS_FOLDER'], file_name)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect('172.20.10.13', username='teo', password='teo')
        print("Connected to SSH")
        stdin, stdout, stderr = ssh.exec_command(record_cmd)
        stdout.channel.recv_exit_status()  # Wait for the command to finish
        print("Record executed")
        if stderr.channel.recv_exit_status() != 0:
            print("Error")
            flash(f"Error recording audio on Raspberry Pi: {stderr.read().decode('utf-8')}", 'error')
            return None
        sftp = ssh.open_sftp()
        sftp.get(file_path, destination)
        sftp.close()
        ssh.close()
        print("ssh closing")

    except Exception as e:
        flash(f"SSH or SCP error: {e}", 'error')
        return None

    return os.path.join(app.config['RECORDINGS_FOLDER'], file_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'email' not in request.form:
        return redirect(request.url)
    files = request.files.getlist('file')
    email_receiver = request.form['email']
    if not files or email_receiver == '':
        return redirect(request.url)

    for file in files:
        if file.filename == '':
            continue
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            analyze_and_send_results(file_path, email_receiver)
    return redirect(url_for('result'))

@app.route('/record', methods=['POST'])
def record():
    if 'duration' not in request.form or 'email' not in request.form:
        return redirect(request.url)
    duration = int(request.form['duration'])
    email_receiver = request.form['email']
    file_name = f"recording_{len(os.listdir(app.config['RECORDINGS_FOLDER'])) + 1}.wav"
    file_path = record_audio(file_name, duration)
    analyze_and_send_results(file_path, email_receiver)
    return redirect(url_for('result'))

@app.route('/record_raspberry', methods=['POST'])
def record_raspberry():
    if 'duration' not in request.form or 'email' not in request.form:
        return redirect(request.url)
    duration = int(request.form['duration'])
    email_receiver = request.form['email']
    file_name = f"raspberry_recording_{len(os.listdir(app.config['RECORDINGS_FOLDER'])) + 1}.wav"
    file_path = record_audio_raspberry_pi(file_name, duration)

    if file_path is None:
        return redirect(url_for('index'))
    analyze_and_send_results(file_path, email_receiver)
    return redirect(url_for('result'))

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
