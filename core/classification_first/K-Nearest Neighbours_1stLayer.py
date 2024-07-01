import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

# Încărcarea datelor
data = pd.read_csv('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\audio_features_all.csv')

# Selectarea featurilor relevante
features = [
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
    'MFCC_11', 'MFCC_12', 'MFCC_13', 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4', 'Chroma_5', 'Chroma_6',
    'Chroma_7', 'Chroma_8', 'Chroma_9', 'Chroma_10', 'Chroma_11', 'Chroma_12', 'SpectralContrast_1',
    'SpectralContrast_2', 'SpectralContrast_3', 'SpectralContrast_4', 'SpectralContrast_5', 'SpectralContrast_6',
    'SpectralContrast_7', 'ZeroCrossingRate'
]

X = data[features]
y = data['label']

# Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scalarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Determinarea numărului optim de vecini folosind cross-validation
best_k = 1
best_score = 0
for k in range(1, 21):  # Testează valori de la 1 la 20
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    avg_score = scores.mean()
    print(f'k={k}, Cross-Validation Accuracy: {avg_score:.4f}')
    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print(f'Cel mai bun număr de vecini: {best_k} cu o acuratețe de cross-validare de {best_score:.4f}')

# Crearea și antrenarea modelului KNN cu numărul optim de vecini
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# Predicțiile pe setul de test
y_pred = knn_model.predict(X_test_scaled)

# Evaluarea modelului
print("KNN - Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Matrice de confuzie:\n", confusion_matrix(y_test, y_pred))


# Asigură-te că directorul pentru salvarea modelului există
model_dir = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'knn_model.joblib')

# Salvarea modelului KNN
dump(knn_model, model_path)