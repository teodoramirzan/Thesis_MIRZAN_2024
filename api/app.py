import os
import joblib
import librosa
import numpy as np
import pandas as pd
import smtplib
import ssl
import wave
import pyaudio
import subprocess
import paramiko
from flask import Flask, request, render_template, redirect, url_for, flash
import pickle
from werkzeug.utils import secure_filename
from email.message import EmailMessage
from config.app_config import Email_App_Password, Email_Username
from tensorflow.keras.models import load_model
from core.models.predict_neuron3 import extract_features, predict_audio
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RECORDINGS_FOLDER'] = r'C:/Users/Matebook 14s/Desktop/Licenta/recordings/PC_recordings'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RECORDINGS_FOLDER'], exist_ok=True)
app.secret_key = 'your_secret_key'

# Load the models
model_rf = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\random_forest_model.joblib')
model_rf_layer2 = joblib.load(
    'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\random_forest_model_level2.joblib')
model_svm = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model.joblib')
model_svm_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model_level2.joblib')
model_knn = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\knn_model.joblib')
model_knn_layer2 = joblib.load('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\knn_model_level2.joblib')
model_nn = load_model('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\audio_classification_model.keras')

# Load the scaler used during training
scaler_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoders used during training
label_encoder_combined_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\label_encoder_combined.pkl'
label_encoder_label_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\label_encoder_label.pkl'
label_encoder_audio_type_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\label_encoder_audio_type.pkl'
with open(label_encoder_combined_path, 'rb') as f:
    label_encoder_combined = pickle.load(f)
with open(label_encoder_label_path, 'rb') as f:
    label_encoder_label = pickle.load(f)
with open(label_encoder_audio_type_path, 'rb') as f:
    label_encoder_audio_type = pickle.load(f)

# Predefined performance metrics for each model
performance_metrics = {
    "Random Forest": {"accuracy": 0.85, "precision": 0.88, "recall": 0.84},
    "SVM": {"accuracy": 0.82, "precision": 0.85, "recall": 0.81},
    "KNN": {"accuracy": 0.80, "precision": 0.83, "recall": 0.79},
    "Neural Network": {"accuracy": 0.90, "precision": 0.92, "recall": 0.89}
}


def is_silent(audio, threshold=0.01):
    rms = librosa.feature.rms(y=audio)[0]
    return np.mean(rms) < threshold


def extract_features_for_prediction(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    if is_silent(audio):
        return None  # Return None or an appropriate value to indicate silence

    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    features.extend(mfccs_processed)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_processed = np.mean(chroma.T, axis=0)
    features.extend(chroma_processed)

    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spec_contrast_processed = np.mean(spec_contrast.T, axis=0)
    features.extend(spec_contrast_processed)

    # Zero Crossing Rate
    zero_crossing = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_processed = np.mean(zero_crossing)
    features.append(zero_crossing_processed)

    feature_names = [
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

    feature_array = np.array(features).reshape(1, -1)
    features_df = pd.DataFrame(feature_array, columns=feature_names)
    features_scaled = scaler.transform(features_df)  # Scale features
    return features_scaled


def predict_rf(file_path):
    features = extract_features_for_prediction(file_path)
    if features is None:
        return "Silent", "None", "Random Forest"

    is_dangerous = model_rf.predict(features)[0]

    if is_dangerous == 'dangerous':
        sound_type = model_rf_layer2.predict(features)[0]
        return "Dangerous", sound_type, "Random Forest"
    return "Not Dangerous", "None", "Random Forest"


def predict_svm(file_path):
    features = extract_features_for_prediction(file_path)
    if features is None:
        return "Silent", "None", "SVM"

    is_dangerous = model_svm.predict(features)[0]

    if is_dangerous == 0:  # Assuming 0 is dangerous, based on previous classification
        sound_type = model_svm_layer2.predict(features)[0]
        return "Dangerous", sound_type, "SVM"
    return "Not Dangerous", "None", "SVM"


def predict_knn(file_path):
    features = extract_features_for_prediction(file_path)
    if features is None:
        return "Silent", "None", "KNN"

    is_dangerous = model_knn.predict(features)[0]

    if is_dangerous == 'dangerous':
        sound_type = model_knn_layer2.predict(features)[0]
        return "Dangerous", sound_type, "KNN"
    return "Not Dangerous", "None", "KNN"


def predict_nn(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Silent", "None", "Neural Network"

    features = scaler.transform([features])
    prediction = model_nn.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label


def send_email(to_email, subject, body):
    em = EmailMessage()
    em['From'] = Email_Username
    em['To'] = to_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(Email_Username, Email_App_Password)
        smtp.send_message(em)


def analyze_and_send_results(file_path, email_receiver, source):
    results = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_rf, type_rf, model_rf_name = predict_rf(file_path)
    results.append((result_rf, type_rf, model_rf_name))

    result_svm, type_svm, model_svm_name = predict_svm(file_path)
    results.append((result_svm, type_svm, model_svm_name))

    result_knn, type_knn, model_knn_name = predict_knn(file_path)
    results.append((result_knn, type_knn, model_knn_name))

    predicted_label = predict_nn(file_path)
    if predicted_label is not None and predicted_label != "Silent":
        decoded_label = label_encoder_combined.inverse_transform(predicted_label)[0]
        label_part = decoded_label // 10
        audio_type_part = decoded_label % 10
        label = label_encoder_label.inverse_transform([label_part])[0]
        audio_type = label_encoder_audio_type.inverse_transform([audio_type_part])[0]
        result_nn = ("Dangerous", audio_type, "Neural Network") if label == "dangerous" else (
            "Not Dangerous", "None", "Neural Network")
        results.append(result_nn)
    else:
        results.append(("Silent", "None", "Neural Network"))

    dangerous_classifications = [(result, sound_type, model_name) for result, sound_type, model_name in results if
                                 result == "Dangerous"]

    if dangerous_classifications:
        short_email_body = "A dangerous sound has been detected.\n"
        send_email(email_receiver, "Sound Danger Detected", short_email_body)

        body_second_email = f"Results of sound analysis:\n\n"
        body_second_email += f"Sample taken at: {current_time}\n"
        body_second_email += f"Recorded using: {source}\n"
        body_second_email += f"Saved at: {file_path}\n\n"

        for result, sound_type, model_name in results:
            metrics = performance_metrics.get(model_name, {})
            body_second_email += f"Model: {model_name}\nResult: {result}\nSound Type: {sound_type}\n"
            body_second_email += f"Accuracy: {metrics.get('accuracy', 'N/A')}\nPrecision: {metrics.get('precision', 'N/A')}\nRecall: {metrics.get('recall', 'N/A')}\n\n"

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
    scp_cmd = f"scp teo@172.20.10.3:{file_path} D:\\{file_name}"
    destination = os.path.join(app.config['RECORDINGS_FOLDER'], file_name)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect('172.20.10.3', username='teo', password='teo')
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
            analyze_and_send_results(file_path, email_receiver, source="upload")
    return redirect(url_for('result'))


@app.route('/record', methods=['POST'])
def record():
    if 'duration' not in request.form or 'email' not in request.form:
        return redirect(request.url)
    duration = int(request.form['duration'])
    email_receiver = request.form['email']
    file_name = f"recording_{len(os.listdir(app.config['RECORDINGS_FOLDER'])) + 1}.wav"
    file_path = record_audio(file_name, duration)
    analyze_and_send_results(file_path, email_receiver, source="laptop")
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
    analyze_and_send_results(file_path, email_receiver, source="raspberry")
    return redirect(url_for('result'))


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
