import os
import requests
import random

# API endpoint
url = "http://127.0.0.1:8000/predict"

def test_api():
    print("=== Démarrage des tests API ===")
    data_dir = "data/val"
    
    if not os.path.exists(data_dir):
        print(f"Erreur: Dossier {data_dir} introuvable.")
        return
        
    classes = os.listdir(data_dir)
    correct = 0
    total = 0
    
    # On teste 2 images aléatoires par classe
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png'))]
        if not images:
            continue
            
        # Sélectionner 2 images au hasard
        sample_images = random.sample(images, min(2, len(images)))
        
        for img_name in sample_images:
            img_path = os.path.join(cls_dir, img_name)
            
            try:
                with open(img_path, "rb") as f:
                    files = {"file": (img_name, f, "image/jpeg")}
                    response = requests.post(url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "success":
                        predicted_class = result.get("primary_diagnosis")
                        confidence = result.get("confidence")
                        
                        is_correct = predicted_class == cls
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        mark = "✅" if is_correct else "❌"
                        print(f"{mark} Vraie classe: {cls} | Prédit: {predicted_class} (Confiance: {confidence}%)")
                        if not is_correct:
                            print(f"   Détails : {result.get('top_3_predictions')}")
                            
                    elif result.get("status") == "uncertain":
                        print(f"❓ Vraie classe: {cls} | Prédit: Incertain (Confiance trop basse)")
                        print(f"   Top prédictions : {result.get('top_predictions')}")
                        total += 1
                    else:
                        print(f"Erreur de l'API: {result}")
                else:
                    print(f"Erreur HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"Erreur lors du test de {img_path}: {e}")

    if total > 0:
        print(f"\n=== Bilan : {correct}/{total} correctes ({correct/total*100:.1f}%) ===")

if __name__ == "__main__":
    test_api()
