import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------

# CONFIGURATION

# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_dir = r"D:\RaviKaGroup\ImageProcessing\try1_2110\OutputProcessed"
batch_size = 32
initial_lr = 1e-4
fine_tune_lr = 1e-5
epochs_stage1 = 5
epochs_stage2 = 5
epochs_stage3 = 5
tta_times = 5  # number of TTA augmentations during validation

# -------------------------

# DATA AUGMENTATION

# -------------------------

train_transform = transforms.Compose([
transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(20),
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])
])

# -------------------------

# SYNTHETIC EXPANSION FOR CLASS 2

# -------------------------

base_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
class_counts = [0] * len(base_dataset.classes)
for _, label in base_dataset.samples:
    class_counts[label] += 1

minority_class = np.argmin(class_counts)
print(f"Minority class detected: {base_dataset.classes[minority_class]} with {class_counts[minority_class]} images")

# Augment class 2 up to 2× majority count

target_count = max(class_counts)
new_samples = []
for img_path, label in base_dataset.samples:
    if label == minority_class:
        for _ in range(int(target_count / class_counts[label]) - 1):
            new_samples.append((img_path, label))
base_dataset.samples.extend(new_samples)

# Split into train/validation (80/20)

num_samples = len(base_dataset)
indices = list(range(num_samples))
split = int(0.8 * num_samples)
random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]
train_set = Subset(base_dataset, train_indices)
val_set = Subset(datasets.ImageFolder(dataset_dir, transform=val_transform), val_indices)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# -------------------------

# MODEL SETUP

# -------------------------

model = models.resnet50(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Sequential(
nn.Linear(num_features, 512),
nn.ReLU(),
nn.Dropout(0.4),
nn.Linear(512, len(base_dataset.classes))
)
model = model.to(device)

# -------------------------

# CLASS WEIGHTS & LOSS

# -------------------------

labels = [label for _, label in base_dataset.samples]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# -------------------------

# TRAINING FUNCTION

# -------------------------

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc.item()

# -------------------------

# VALIDATION FUNCTION (TTA)

# -------------------------

def validate_model(model, dataloader, criterion, use_tta=False):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if use_tta:
                outputs_tta = torch.zeros((inputs.size(0), len(base_dataset.classes))).to(device)
                for _ in range(tta_times):
                    aug_inputs = torch.stack([train_transform(img.cpu()) for img in inputs.cpu()])
                    aug_inputs = aug_inputs.to(device)
                    outputs_tta += torch.softmax(model(aug_inputs), dim=1)
                    outputs = outputs_tta / tta_times
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item(), all_labels, all_preds

# -------------------------

# PROGRESSIVE UNFREEZING

# -------------------------

def unfreeze_layers(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True

# Freeze all initially

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Stage 1: Train classifier head

print("\nStage 1: Training top layers...")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)
for epoch in range(epochs_stage1):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = validate_model(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{epochs_stage1} - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Stage 2: Unfreeze layer4

print("\nStage 2: Fine-tuning layer4...")
unfreeze_layers(model, "layer4")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr)
for epoch in range(epochs_stage2):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = validate_model(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{epochs_stage2} - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Stage 3: Unfreeze layer3

print("\nStage 3: Fine-tuning layer3...")
unfreeze_layers(model, "layer3")
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr)
for epoch in range(epochs_stage3):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
    val_loss, val_acc, y_true, y_pred = validate_model(model, val_loader, criterion, use_tta=True)
    print(f"Epoch {epoch+1}/{epochs_stage3} - Train Acc: {train_acc:.4f} | Val Acc (TTA): {val_acc:.4f}")

# -------------------------

# EVALUATION

# -------------------------

print("\nConfusion Matrix & Classification Report:")
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=base_dataset.classes)
print(report)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
xticklabels=base_dataset.classes,
yticklabels=base_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (with TTA)")
plt.show()

torch.save(model.state_dict(), "final_resnet50_finetuned.pth")
print("\nModel saved as 'final_resnet50_finetuned.pth'")
