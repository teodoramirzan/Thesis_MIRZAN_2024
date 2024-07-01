import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pickle

# Load the data
data_path = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\All_preprocessed_v2\\audio_features_all.csv"
data = pd.read_csv(data_path)

# Split features and labels
X = data.drop(columns=['label', 'audio_type'])
y = data[['label', 'audio_type']].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Encode the labels and audio_type
label_encoder_label = LabelEncoder()
label_encoder_audio_type = LabelEncoder()

y['label'] = label_encoder_label.fit_transform(y['label'])
y['audio_type'] = label_encoder_audio_type.fit_transform(y['audio_type'])

# Create a combined label column for multi-class classification
y_combined = y['label'] * 10 + y['audio_type']

# Normalize the combined labels using LabelEncoder
label_encoder_combined = LabelEncoder()
y_combined = label_encoder_combined.fit_transform(y_combined)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_combined, test_size=0.2, random_state=42)

# Determine the number of classes
num_classes = len(np.unique(y_combined))

# Construirea modelului neuronal
model = Sequential()
model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # ajustarea numărului de clase

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict and evaluate the detailed performance
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_labels))

# Save the model in the native Keras format
model_save_dir = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

model_save_path = os.path.join(model_save_dir, "audio_classification_model.keras")
model.save(model_save_path)

print(f"Modelul a fost salvat la: {model_save_path}")

# Save the scaler
scaler_save_path = os.path.join(model_save_dir, "scaler.pkl")
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Scalerul a fost salvat la: {scaler_save_path}")

# Save the combined label encoder
label_encoder_combined_path = os.path.join(model_save_dir, "label_encoder_combined.pkl")
with open(label_encoder_combined_path, 'wb') as f:
    pickle.dump(label_encoder_combined, f)

print(f"Label encoder combined a fost salvat la: {label_encoder_combined_path}")

# Save individual label encoders
label_encoder_label_path = os.path.join(model_save_dir, "label_encoder_label.pkl")
label_encoder_audio_type_path = os.path.join(model_save_dir, "label_encoder_audio_type.pkl")
with open(label_encoder_label_path, 'wb') as f:
    pickle.dump(label_encoder_label, f)
with open(label_encoder_audio_type_path, 'wb') as f:
    pickle.dump(label_encoder_audio_type, f)

print(f"Label encoder label a fost salvat la: {label_encoder_label_path}")
print(f"Label encoder audio type a fost salvat la: {label_encoder_audio_type_path}")
