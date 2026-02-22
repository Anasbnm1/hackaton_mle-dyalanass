import cv2
import numpy as np

def is_image_blurry(image_bytes: bytes, threshold: float = 100.0) -> bool:
    """
    Vérifie si une image est trop floue pour être analysée.
    Utilise la variance du Laplacien pour estimer la netteté.
    """
    try:
        # Convertir les bytes de l'image en tableau numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return True
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculer la variance du Laplacien
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Si la variance est inférieure au seuil, l'image est considérée comme floue
        return variance < threshold
    except Exception as e:
        print(f"Erreur lors de l'analyse du flou : {e}")
        return True # En cas d'erreur, on rejette par prudence

# Dictionnaire de conseils basiques (MVP)
conseils_plantes = {
    "Tomato_healthy": [
        "Continuez l'arrosage régulier.",
        "Vérifiez l'exposition au soleil.",
        "Aucune action spécifique requise."
    ],
    "Tomato_Late_blight": [
        {"icon": "✂️", "text": "Retirez immédiatement les feuilles noircies et détruisez-les."},
        {"icon": "💨", "text": "Aérez au maximum pour faire baisser l'humidité."},
        {"icon": "🛡️", "text": "Traitez avec un fongicide doux (ex: bouillie bordelaise)."}
    ],
    "Tomato_Leaf_Mold": [
        {"icon": "🌡️", "text": "Baissez l'humidité ambiante, ventilez abondamment la serre ou la plante."},
        {"icon": "✂️", "text": "Taillez les feuilles basses atteintes pour améliorer la circulation de l'air."},
        {"icon": "💧", "text": "Évitez formellement d'arroser le feuillage."}
    ],
    "Tomato_Early_blight": [
        "Retirez les feuilles infectées du bas de la plante.",
        "Améliorez la circulation de l'air entre les plants.",
        "Essayez une rotation des cultures l'année prochaine."
    ],
    "Potato_healthy": [
        "Maintenez de bonnes conditions de culture.",
        "Surveillez les ravageurs courants comme les doryphores."
    ],
    "Potato_Late_blight": [
        "Détruisez les parties infectées (ne les compostez pas).",
        "Traitez rapidement avec un fongicide spécifique.",
        "Si l'infection est grave, récoltez ce qui peut l'être."
    ]
}

def get_advice(class_name: str) -> list:
    """Retourne une liste de conseils pour une classe donnée."""
    return conseils_plantes.get(class_name, [
        "Observer la plante régulièrement.",
        "Isoler la plante en cas de doute.",
        "Consulter un expert ou un forum spécialisé si les symptômes s'aggravent."
    ])
