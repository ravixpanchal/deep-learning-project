# ==============================================
# MULTICLASS IMAGE CLASSIFIER WITH SYNTHETIC UPSAMPLING
# ==============================================
import os
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Transpose, RandomBrightnessContrast,
    HueSaturationValue, CLAHE, RandomGamma, Blur, GaussNoise, ElasticTransform,
    GridDistortion, OpticalDistortion, RandomResizedCrop, ShiftScaleRotate
)

from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --------------------------
# CONFIGURATION
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = 'D:/RaviKaGroup/ImageProcessing/try1_2110/OutputProcessed'
aug_class2_dir = os.path.join(base_dir, 'output_jpg2_aug')
num_classes = 3
batch_size = 32
num_epochs = 40
lr = 1e-4
cutmix_alpha = 1.0
label_smooth = 0.1

# ---------------------------------------------------------
# STEP 1: SYNTHETICALLY AUGMENT CLASS 2 TO ~300 IMAGES
# ---------------------------------------------------------
os.makedirs(aug_class2_dir, exist_ok=True)
class2_dir = os.path.join(base_dir, 'output_jpg2')
existing_imgs = [f for f in os.listdir(class2_dir) if f.lower().endswith(('.jpg','.png'))]
n_existing = len(existing_imgs)
target_n = 300
to_generate = target_n - n_existing
print(f"Generating {to_generate} new synthetic images for class 2...")

augment = Compose([
    RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.3),
    RandomRotate90(p=0.5),
    Transpose(p=0.5),
    RandomBrightnessContrast(p=0.4),
    HueSaturationValue(p=0.3),
    CLAHE(p=0.3),
    RandomGamma(p=0.3),
    Blur(p=0.2),
    GaussNoise(p=0.2),
    ElasticTransform(p=0.2),
    GridDistortion(p=0.2),
    OpticalDistortion(p=0.2),
    ShiftScaleRotate(p=0.3)
])

for i in tqdm(range(to_generate)):
    img_name = existing_imgs[i % n_existing]
    img_path = os.path.join(class2_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = augment(image=img)
    out_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
    out_name = f"aug_{i}_{img_name}"
    cv2.imwrite(os.path.join(aug_class2_dir, out_name), out_img)

print("Synthetic augmentation complete.\n")

# --------------------------
# DATA AUGMENTATION
# --------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomErasing(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --------------------------
# DATASETS
# --------------------------
dataset = datasets.ImageFolder(base_dir, transform=train_transform)
class_names = dataset.classes
targets = np.array(dataset.targets)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(np.zeros(len(targets)), targets):
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(
        datasets.ImageFolder(base_dir, transform=val_transform), val_idx)

# --------------------------
# CLASS WEIGHTS
# --------------------------
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(targets),
                                     y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)

# --------------------------
# DATA LOADERS (balanced)
# --------------------------
sample_weights = [class_weights[label] for label in targets[train_idx]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# --------------------------
# MODEL (EfficientNet-B0)
# --------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --------------------------
# CUTMIX FUNCTION
# --------------------------
def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --------------------------
# TRAINING LOOP
# --------------------------
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=cutmix_alpha)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cutmix_criterion(outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (lam * preds.eq(targets_a).sum().item() +
                    (1 - lam) * preds.eq(targets_b).sum().item())

    scheduler.step()
    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = 100 * correct / total
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_effnetb0.pth')

print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
model.load_state_dict(torch.load('best_model_effnetb0.pth'))
model.eval()
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
