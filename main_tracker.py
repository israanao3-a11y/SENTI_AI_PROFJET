import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

from gaze_tracker_utils import compute_actual_eye_gaze
from gaze_logger import init_csv, append_gaze_data
from gaze_visualizer import draw_gaze_bar

# =====================================================================
# 1. RECONSTRUCTION DU MODELE (METHODE DE YOUSRA)
# =====================================================================
def build_model():
    """Reconstruit la structure du reseau CNN pour eviter les conflits d'importation"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

# Initialisation du modele et chargement des poids
model = build_model()
try:
    # Chargement des poids uniquement pour eviter le bug de l'argument 'lr'
    model.load_weights('fer2013_emotion_model.h5')
    print("[SUCCES] Poids du modele charges avec succes.")
except Exception:
    # Methode de secours sans compilation si load_weights echoue
    model = tf.keras.models.load_model('fer2013_emotion_model.h5', compile=False)
    print("[SUCCES] Modele charge via secours (compile=False).")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48

# =====================================================================
# * CONFIGURATION DU LISSAGE (STABILISATION DES POURCENTAGES)
# =====================================================================
predictions_history = []  # Memoire pour stocker les dernieres frames et calculer la moyenne

# Seuils pour le suivi du regard (Gaze Thresholds)
FIXATION_THRESHOLD = 0.05
LEFT_THRESHOLD = -0.2
RIGHT_THRESHOLD = 0.2

# =====================================================================
# 2. INITIALISATION DE MEDIAPIPE FACE MESH
# =====================================================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Preparation du fichier de log CSV
csv_file = 'gaze_data.csv'
init_csv(csv_file)

# Initialisation de la camera (CAP_DSHOW est optimise pour Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
prev_time = 0
frame_count = 0
last_side = None
switch_count = 0

gaze_counts = {'Left': 0, 'Center': 0, 'Right': 0}

# Variables de controle de l'affichage
show_face = True
show_eyes_mouth = False
show_rectangle = True
show_gaze = True

# =====================================================================
# 3. BOUCLE PRINCIPALE ET TRAITEMENT TEMPS REEL
# =====================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible d'acceder a la camera.")
        break
        
    frame = cv2.flip(frame, 1)  # Effet miroir pour plus de naturel
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detection des points du visage via MediaPipe
    face_results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
        
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Affichage du maillage vert complet
            if show_face:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec)
            # Affichage alternatif restreint (yeux et bouche)
            elif show_eyes_mouth:
                for idx in list(range(33, 133)) + list(range(263, 294)) + list(range(78, 88)):
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                    
            # Definition de la zone d'interet du visage (ROI) pour la prediction
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            
            x_min = int(min(x_coords) * w)
            y_min = int(min(y_coords) * h)
            x_max = int(max(x_coords) * w)
            y_max = int(max(y_coords) * h)
            
            # Securite pour ne pas depasser les dimensions de l'image
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size > 0:
                # Preprocessing de l'image pour le modele (Gris, Resize, Normalisation)
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                normalized_face = resized_face / 255.0
                face_input = normalized_face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                
                # 1. Prediction brute instantanee
                preds = model.predict(face_input, verbose=0)[0]
                
                # 2. Ajout de la prediction a l'historique pour le lissage
                predictions_history.append(preds)
                if len(predictions_history) > 10:  # On garde les 10 dernieres frames pour stabiliser
                    predictions_history.pop(0)
                
                # 3. Calcul de la moyenne pour eliminer les sauts brusques
                smoothed_preds = np.mean(predictions_history, axis=0)
                
                emotion_idx = np.argmax(smoothed_preds)
                emotion_text = emotion_labels[emotion_idx]
                pourcentage = smoothed_preds[emotion_idx] * 100
                
                # Affichage du rectangle bleu et du texte lisse
                if show_rectangle:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    texte_affichage = f"{emotion_text} : {pourcentage:.1f}%"
                    cv2.putText(frame, texte_affichage, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
            # Estimation et suivi de l'orientation du regard
            if show_gaze:
                eye_center, gaze_vec = compute_actual_eye_gaze(face_landmarks.landmark, w, h)
                eye_tip = tuple(np.int32(eye_center + gaze_vec * 80))
                cv2.arrowedLine(frame, tuple(np.int32(eye_center)), eye_tip, (0, 255, 0), 2)
                
                gaze_x = gaze_vec[0]
                fixation = abs(gaze_x) < FIXATION_THRESHOLD
                
                side = "Center"
                if gaze_x < LEFT_THRESHOLD:
                    side = 'Left'
                elif gaze_x > RIGHT_THRESHOLD:
                    side = 'Right'
                    
                gaze_counts[side] += 1
                
                if last_side and side != last_side:
                    switch_count += 1
                last_side = side
                
                # Enregistrement des donnees dans le fichier CSV
                append_gaze_data(csv_file, frame_count, gaze_vec, fixation, side)
                
    # Rendu de la barre graphique superieure pour le regard
    draw_gaze_bar(frame, gaze_counts)
    cv2.putText(frame, f'Switches: {switch_count}', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
    
    # Calcul et affichage du FPS (Images par seconde)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    frame_count += 1
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Autism-related Emotion and Behavior Detection', frame)
    
    # Gestion des touches du clavier pour l'interaction
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quitter
        break
    elif key == ord('f'):  # Mode maillage complet (Full face)
        show_face = True
        show_eyes_mouth = False
    elif key == ord('m'):  # Mode restreint (Mouth and eyes)
        show_face = False
        show_eyes_mouth = True
    elif key == ord('r'):  # Afficher/Masquer le rectangle bleu
        show_rectangle = not show_rectangle
    elif key == ord('g'):  # Afficher/Masquer la fleche du regard
        show_gaze = not show_gaze

# Nettoyage et fermeture des fenetres
cap.release()
cv2.destroyAllWindows()
print("[INFO] Application terminee proprement.")