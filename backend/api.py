import io
import os
import cv2
import base64
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

load_dotenv() # Charge le fichier .env

# Import pytorch-grad-cam pour l'explicabilité (Bonus)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Imports pour le ChatBot IA (Gemini)
import google.generativeai as genai
from pydantic import BaseModel

from utils import is_image_blurry, get_advice

# ==========================================
# CONFIGURATION DE L'API
# ==========================================
app = FastAPI(
    title="PlantDoc Vision API", 
    description="API de diagnostic de santé des plantes par image (MVP)",
    version="1.0.0"
)

# Autoriser le Frontend (React/Vue/Angular) à communiquer avec l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales du modèle
MODEL_PATH = "plantdoc_mobilenetv2.pth"
CLASSES_PATH = "classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
classes = []
cam = None # Grad-CAM instance

# Pré-traitement identique à l'entraînement (sans data-augmentation)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_model():
    global model, classes, cam
    
    # 1. Charger les classes
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        # Fallback pour la démo si non entraîné
        classes = ["Tomato_healthy", "Tomato_Late_blight", "Tomato_Early_blight", "Potato_healthy", "Potato_Late_blight", "Tomato_Leaf_Mold"]
        print("Info : Utilisation des classes par défaut.")
    
    # 2. Charger l'architecture MobileNetV2
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    
    # 3. Charger les poids (si disponibles)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Modèle chargé avec succès sur {DEVICE}.")
    else:
        print("ATTENTION : Fichier de poids '.pth' non trouvé. Le modèle fera des prédictions aléatoires.")
        
    model = model.to(DEVICE)
    model.eval() # Mode inférence

    # 4. Initialisation de Grad-CAM sur la dernière couche convolutive
    # Sur MobileNetV2, la dernière couche "features" avant le classifier
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    """
    Endpoint principal : Analyse une photo de feuille et renvoie un diagnostic.
    """
    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="Veuillez uploader un fichier image valide.")
         
    image_bytes = await file.read()
    
    # --- CONTRÔLE QUALITÉ ---
    if is_image_blurry(image_bytes, threshold=100.0):
        return {
            "status": "error", 
            "message": "L'image est trop floue. Veuillez prendre une photo plus nette pour un bon diagnostic."
        }

    # --- LECTURE & PRÉ-TRAITEMENT ---
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image. Fichier corrompu ?")
        
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # --- INFÉRENCE ---
    with torch.no_grad():
        output = model(input_tensor)
        # Convertir les logits en probabilités (softmax)
        probabilities = F.softmax(output, dim=1)[0]
        
    # Extraire le Top-1 (la meilleure probabilité)
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    
    top1_prob_list = top1_prob.cpu().numpy().tolist()
    top1_catid_list = top1_catid.cpu().numpy().tolist()
    
    best_confidence = top1_prob_list[0]
    best_class = classes[top1_catid_list[0]]
    
    # --- GESTION DE L'INCERTITUDE ---
    # Si la confiance max est inférieure à 50%
    if best_confidence < 0.50:
        return {
            "status": "uncertain",
            "message": "Je ne sais pas / Confiance trop basse. L'image pourrait ne pas être une feuille claire des plantes connues.",
            "top_predictions": [
                {"class": best_class, "confidence": round(best_confidence * 100, 2)} 
            ]
        }

    # --- RÉPONSE FORMATÉE ---
    results = [
        {"class": best_class, "confidence": round(best_confidence * 100, 2)}
    ]

    advice = get_advice(best_class)

    return {
        "status": "success",
        "primary_diagnosis": best_class,
        "confidence": round(best_confidence * 100, 2),
        "top_3_predictions": results,
        "advice": advice
    }

@app.post("/explain")
async def explain_prediction(file: UploadFile = File(...)):
    """
    Génère une Heatmap Grad-CAM pour montrer où le modèle 'regarde'.
    Renvoie l'image traitée en base64 pour être affichée directement par le frontend.
    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Préparer l'image pour l'affichage (taille de l'entrée du modèle)
        img_resized = image.resize((224, 224))
        rgb_img = np.float32(img_resized) / 255
        
        # Pré-traitement et tenseur
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Générer la heatmap avec Grad-CAM (cible = None -> prend la classe prédite Max)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :] # On prend la première image du batch
        
        # Superposition avec l'image d'origine
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        pil_vis = Image.fromarray(visualization)
        
        # Encodage en Base64
        buffered = io.BytesIO()
        pil_vis.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "heatmap_base64": f"data:image/jpeg;base64,{img_str}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Erreur de génération heatmap: {str(e)}"}

# ==========================================
# CHATBOT INTELLIGENT (LLM)
# ==========================================

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_bot(req: ChatRequest):
    """
    Assistant IA expert en botanique. 
    Fonctionne avec l'API Google Gemini si GEMINI_API_KEY est dans les variables d'environnement.
    """
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Le message est vide.")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    # 1. Si aucune clé API n'est configurée, on utilise un comportement de démonstration
    if not api_key:
        return {
            "status": "success",
            "response": "Bonjour ! Le mode IA complet est désactivé. Veuillez configurer `GEMINI_API_KEY` côté Backend pour que je puisse répondre intelligemment."
        }

    # 2. Si la clé est présente, on interroge Gemini
    try:
        genai.configure(api_key=api_key)
        # L'API Key de l'utilisateur ne supporte que les tout derniers modèles comme 2.5
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Le System Prompt pour forcer l'IA à rester dans son rôle
        system_prompt = (
            "Tu es 'Dr Plant', un expert agronome et botaniste. "
            "Ton but est d'aider les utilisateurs avec leurs plantes de manière chaleureuse, courte et concise ("
            "maximum 3 phrases). N'utilise jamais de markdown complexe, garde le texte pur. "
            "Si la question ne concerne pas du tout la nature, les plantes ou l'agriculture, dis poliment que tu n'es qu'un expert végétal.\n\n"
            f"Utilisateur: {user_message}"
        )

        response = model.generate_content(system_prompt)
        ai_text = response.text.replace('*', '') # Nettoyage basique

        return {
            "status": "success",
            "response": ai_text
        }
        
    except Exception as e:
        print(f"Erreur LLM : {str(e)}")
        return {
            "status": "error",
            "response": "Désolé, j'éprouve des difficultés de connexion avec mon cerveau IA actuellement."
        }

# ==========================================
# INSTRUCTIONS D'EXÉCUTION (POUR LE DEV)
# ==========================================
# 1. pip install -r requirements.txt
# 2. uvicorn api:app --reload
# 3. L'API tourne sur: http://127.0.0.1:8000
# 4. Docs interactives: http://127.0.0.1:8000/docs
