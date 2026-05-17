# download_model.py
import urllib.request
import os

print("=== Téléchargement du modèle émotions en cours... ===")

url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
output_name = "fer2013_emotion_model.h5"

try:
    urllib.request.urlretrieve(url, output_name)
    if os.path.exists(output_name):
        print(f"[Succès] Le modèle a été téléchargé et enregistré sous : {output_name}")
    else:
        print("[Erreur] Le fichier n'a pas pu être créé.")
except Exception as e:
    print(f"[Erreur lors du téléchargement] : {e}")
    