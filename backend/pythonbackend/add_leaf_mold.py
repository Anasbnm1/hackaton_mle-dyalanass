import os
import shutil
import random

source = r"c:\Users\anasb\Desktop\PlantDoc\backend\Kaggle_PlantVillage\Tomato_Leaf_Mold"
train_dir = r"c:\Users\anasb\Desktop\PlantDoc\backend\data\train\Tomato_Leaf_Mold"
val_dir = r"c:\Users\anasb\Desktop\PlantDoc\backend\data\val\Tomato_Leaf_Mold"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

files = [f for f in os.listdir(source) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
random.seed(42)
random.shuffle(files)

split_index = int(0.8 * len(files))
train_files = files[:split_index]
val_files = files[split_index:]

for f in train_files:
    shutil.copy(os.path.join(source, f), os.path.join(train_dir, f))

for f in val_files:
    shutil.copy(os.path.join(source, f), os.path.join(val_dir, f))

print(f"Added Tomato_Leaf_Mold: Copied {len(train_files)} to train, {len(val_files)} to val.")
