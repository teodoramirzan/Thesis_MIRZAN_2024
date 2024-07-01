import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from joblib import load
from sklearn.model_selection import train_test_split


# Calea spre modelul salvat
model_path = 'C:\\Users\\Matebook 14s\\Desktop\\Licenta\\saved_models\\random_forest_model.joblib'

# Încărcarea modelului Random Forest
model = load(model_path)

# Încărcarea datelor de test, presupunând că acest cod rulează separat și trebuie să reîncarcăm datele
# Dacă deja ai X_test și y_test disponibile, poți sări peste această încărcare
data = pd.read_csv('C:\\Users\\Matebook 14s\\Desktop\\Licenta\\data_test\\audio_features_all.csv')
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Predicțiile pe setul de test, presupunând că setul de test este același sau a fost reîncărcat
predictions = model.predict(X_test)

# Evaluarea modelului
print("Accuracy on Test Set:", accuracy_score(y_test, predictions))
print("Classification Report on Test Set:\n", classification_report(y_test, predictions))
