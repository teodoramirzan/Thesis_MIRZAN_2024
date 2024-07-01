import requests

client_id = 'MswkwrNKlY4vph5X0wbO' #MswkwrNKlY4vph5X0wbO
client_secret = 'K47hnHyyr7aIqIO8RcMa1hEgofudTBV11SsfRf49' #K47hnHyyr7aIqIO8RcMa1hEgofudTBV11SsfRf49
code = 't9DWwEzrjkKNBtukVliKQqhdop7g06' #AUTHORIZATION_CODE_RECEIVED_FROM_API

# Endpoint-ul pentru obținerea token-ului de acces
token_url = 'http://freesound.org/apiv2/oauth2/access_token/'

# Parametrii pentru cererea POST pentru a obține token-ul de acces
data = {
    'client_id': client_id,
    'client_secret': client_secret,
    'grant_type': 'authorization_code',
    'code': code
}

# Faceți cererea și obțineți răspunsul
response = requests.post(token_url, data=data)

# Verificați dacă cererea a fost reușită
if response.status_code == 200:
    # Extragerea token-ului de acces din răspuns
    access_token = response.json()['access_token']
    print("Token de acces obținut:", access_token)
else:
    print("Eroare la obținerea token-ului de acces:", response.status_code)
