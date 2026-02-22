import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================
# CONFIGURATION AMÉLIORÉE (Modèle plus robuste)
# ==========================================
DATA_DIR = 'data' 
BATCH_SIZE = 32
EPOCHS = 15  # On augmente un peu le temps d'entraînement
LEARNING_RATE = 0.0005 # Learning rate plus petit pour le fine-tuning
UNFREEZE_BLOCKS = 3 # Nombre de blocs convolutifs à dégeler à la fin de MobileNetV2

def get_data_loaders():
    # 1. Pré-traitement et Data Augmentation PLUS ROBUSTE pour les vraies photos
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Simulation de zoom
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Les feuilles peuvent être prises dans n'importe quel sens
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Variations de lumière importantes
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Simulation de léger flou caméra
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms),
        'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transforms)
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    
    return dataloaders, image_datasets['train'].classes

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Entraînement Avancé sur {device} ===")

    try:
        dataloaders, class_names = get_data_loaders()
        num_classes = len(class_names)
        print(f"Classes détectées: {class_names}")
    except FileNotFoundError:
        print("Erreur: Le dossier 'data/train' ou 'data/val' est introuvable.")
        return

    # 2. Modèle Transfer Learning (MobileNetV2)
    # weights=models.MobileNet_V2_Weights.DEFAULT est la nouvelle syntaxe recommandée
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        print("Poids ImageNet chargés par défaut pour le Transfer Learning.")
    except Exception:
        model = models.mobilenet_v2(pretrained=True)
    
    # 3. Stratégie de Fine-Tuning (Dégel partiel)
    # Au lieu de tout geler, on gèle le début, et on laisse les dernières couches apprendre les spécificités des feuilles
    for param in model.parameters():
        param.requires_grad = False
        
    # On dégèle les derniers blocs de l'extracteur de features
    for idx, child in enumerate(model.features):
        if idx >= len(model.features) - UNFREEZE_BLOCKS:
            for param in child.parameters():
                param.requires_grad = True
                
    # Remplacer et dégeler le classifieur final
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    # 4. Optimiseur et Loss
    criterion = nn.CrossEntropyLoss()
    # On optimise uniquement les paramètres qui nécessitent un gradient (classifier + derniers blocs)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Scheduler pour réduire le learning rate si le modèle stagne
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 5. Boucle d'entraînement
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                scheduler.step(epoch_acc) # Ajuster le LR basé sur la précision de validation
                
                # Sauvegarder uniquement si on bat le record
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'plantdoc_mobilenetv2.pth')
                    print(f"🌟 Nouveau meilleur modèle sauvegardé ! (Précision: {best_acc:.4f})")

    print(f"\n✅ Entraînement terminé. Meilleure V-Acc: {best_acc:.4f}.")
    
    with open("classes.txt", "w") as f:
        f.write("\n".join(class_names))

if __name__ == '__main__':
    train_model()
