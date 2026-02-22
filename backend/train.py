import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# CONFIGURATION (MVP 48h - 5 classes cibles)
# ==========================================
DATA_DIR = 'data' # Dossier contenant 'train' et 'val'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 5

def train_model():
    """
    Pipeline de préparation et d'entraînement avec PyTorch (MobileNetV2).
    Utilise le Transfer Learning pour un prototypage rapide et léger.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Entraînement sur {device} ===")

    # 1. Pré-traitement et Data Augmentation
    # MobileNetV2 attend des images RGB de taille 224x224 normalisées
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # Resize & Crop
            transforms.RandomHorizontalFlip(), # Flip
            transforms.RandomRotation(20),     # Rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Zoom / variations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalisation standard ImageNet
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # Charger les datasets
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
                       for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(f"Classes détectées: {class_names}")
    except FileNotFoundError:
        print("Erreur: Le dossier 'data/train' ou 'data/val' est introuvable.")
        print("Pour lancer l'entraînement réel, placez le dataset 'PlantVillage' (réduit à 5 classes) dans le dossier 'data/'.")
        return

    # 2. Modèle Transfer Learning (MobileNetV2)
    model = models.mobilenet_v2(pretrained=True)
    
    # Geler les couches de base pour préserver les poids ImageNet initiaux
    for param in model.parameters():
        param.requires_grad = False
        
    # Remplacer le classifieur final pour s'adapter à nos `NUM_CLASSES`
    # (le classifier d'origine termine par Linear(1280, 1000))
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model = model.to(device)

    # 3. Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    # On optimise uniquement les paramètres du classifieur remplaçant
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # 4. Boucle d'entraînement
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Sauvegarder le meilleur modèle
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'plantdoc_mobilenetv2.pth')

    print(f"Entraînement terminé. Meilleure V-Acc: {best_acc:.4f}. Modèle sauvegardé ('plantdoc_mobilenetv2.pth').")
    
    # Sauvegarder la liste des classes pour y accéder depuis l'API
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))

if __name__ == '__main__':
    train_model()
