import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Calea către fișierul CSV cu caracteristicile audio
file_path = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\All_preprocessed_v2\\audio_features_all.csv"

# Încarcă datele din fișierul CSV
data = pd.read_csv(file_path)

# Verifică primele rânduri din date pentru a înțelege structura acestora
print(data.head())

# Extrage caracteristicile (features) și etichetele (labels)
X = data.drop(columns=['label', 'audio_type'])
y = data['label']

# Convertim etichetele (labels) în format numeric dacă nu sunt deja
y = y.astype('category')
print(y.cat.categories)  # Verifică ordinea etichetelor

# Convertim etichetele (labels) în coduri numerice
y = y.cat.codes

# One-hot encoding pentru caracteristicile categorice (dacă există)
X = pd.get_dummies(X)

# Împarte datele în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creează și antrenează modelul SVM
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

# Salvează modelul antrenat într-un fișier
model_path = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\old\\svm_model.joblib"
joblib.dump(model_svm, model_path)

# Evaluează modelul
y_pred = model_svm.predict(X_test)
print("Raport de clasificare:\n", classification_report(y_test, y_pred, target_names=['not dangerous', 'dangerous']))
print("Matrice de confuzie:\n", confusion_matrix(y_test, y_pred))
print(f"Acuratețea modelului: {model_svm.score(X_test, y_test) * 100:.2f}%")

print("Modelul SVM a fost antrenat și salvat la:", model_path)
