import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Calea către fișierul CSV cu caracteristicile audio
file_path = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\All_preprocessed_v2\\audio_features_all.csv"

# Încarcă datele din fișierul CSV
data = pd.read_csv(file_path)

# Verifică primele rânduri din date pentru a înțelege structura acestora
print(data.head())

# Selectăm doar rândurile care sunt periculoase
dangerous_data = data[data['label'] == 'dangerous']

# Verifică dacă există date periculoase
if dangerous_data.empty:
    print("Nu există date periculoase în setul de date.")
else:
    # Extrage caracteristicile (features) și etichetele (audio_type)
    X = dangerous_data.drop(columns=['label', 'audio_type'])
    y = dangerous_data['audio_type']

    # Convertim etichetele (audio_type) în format numeric dacă nu sunt deja
    y = y.astype('category').cat.codes

    # One-hot encoding pentru caracteristicile categorice (dacă există)
    X = pd.get_dummies(X)

    # Împarte datele în seturi de antrenament și testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verificăm dacă seturile de antrenament și testare conțin date
    print(f"Numărul de exemple de antrenament: {len(X_train)}")
    print(f"Numărul de exemple de test: {len(X_test)}")

    # Creează și antrenează modelul Random Forest
    model_rf_layer2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf_layer2.fit(X_train, y_train)

    # Salvează modelul antrenat într-un fișier
    model_path_layer2 = "C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model_2nd.joblib"
    joblib.dump(model_rf_layer2, model_path_layer2)

    # Evaluează modelul
    y_pred_layer2 = model_rf_layer2.predict(X_test)
    print("Raport de clasificare pentru nivelul 2:\n", classification_report(y_test, y_pred_layer2))
    print("Matrice de confuzie pentru nivelul 2:\n", confusion_matrix(y_test, y_pred_layer2))
    print(f"Acuratețea modelului pentru nivelul 2: {model_rf_layer2.score(X_test, y_test) * 100:.2f}%")

    print("Modelul Random Forest pentru nivelul 2 a fost antrenat și salvat la:", model_path_layer2)
