import matplotlib.pyplot as plt
import networkx as nx

# Creează un graf direcționat
G = nx.DiGraph()

# Adaugă noduri cu etichete în română
noduri = {
    "Start": {"shape": "circle", "color": "#cdc2cf"},
    "Verificare sursă: freesound sau kaggle?": {"shape": "diamond", "color": "#c2d1f0"},
    "Descărcare sunete de la sursă": {"shape": "box", "color": "#f0c2c2"},
    "Redenumirea automată a fișierelor": {"shape": "box", "color": "#f0c2c2"},
    "Preprocesarea sunetelor": {"shape": "box", "color": "#f0c2c2"},
    "Extracția caracteristicilor și crearea fișierului CSV": {"shape": "box", "color": "#f0c2c2"},
    "Antrenarea modelelor și salvarea lor": {"shape": "box", "color": "#f0c2c2"},
    "Transferul sunetelor de pe Raspberry Pi pe laptop": {"shape": "box", "color": "#f0c2c2"},
    "Predicție pe baza sunetelor înregistrate": {"shape": "box", "color": "#f0c2c2"},
    "Generarea metricilor de performanță și a matricilor de confuzie": {"shape": "box", "color": "#f0c2c2"},
    "Integrarea cu interfața HTML": {"shape": "box", "color": "#f0c2c2"},
    "Testarea finală a sistemului": {"shape": "box", "color": "#f0c2c2"},
    "End": {"shape": "circle", "color": "#cdc2cf"}
}

for i, (nod, attrs) in enumerate(noduri.items()):
    G.add_node(nod, shape=attrs["shape"], color=attrs["color"], label=nod)

# Adaugă muchii
muchii = [
    ("Start", "Verificare sursă: freesound sau kaggle?"),  # Start -> Verificare sursă
    ("Verificare sursă: freesound sau kaggle?", "Descărcare sunete de la sursă"),  # Verificare sursă -> Descărcare sunete (Da)
    ("Verificare sursă: freesound sau kaggle?", "Redenumirea automată a fișierelor"),  # Verificare sursă -> Redenumire automată (Nu)
    ("Descărcare sunete de la sursă", "Redenumirea automată a fișierelor"),  # Descărcare sunete -> Redenumire automată
    ("Redenumirea automată a fișierelor", "Preprocesarea sunetelor"),  # Redenumire automată -> Preprocesarea sunetelor
    ("Preprocesarea sunetelor", "Extracția caracteristicilor și crearea fișierului CSV"),  # Preprocesarea sunetelor -> Extracția caracteristicilor
    ("Extracția caracteristicilor și crearea fișierului CSV", "Antrenarea modelelor și salvarea lor"),  # Extracția caracteristicilor -> Antrenarea modelelor
    ("Antrenarea modelelor și salvarea lor", "Transferul sunetelor de pe Raspberry Pi pe laptop"),  # Antrenarea modelelor -> Transferul sunetelor
    ("Transferul sunetelor de pe Raspberry Pi pe laptop", "Predicție pe baza sunetelor înregistrate"),
]