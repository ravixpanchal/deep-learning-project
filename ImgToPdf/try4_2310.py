import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
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
tta_times = 5

# -------------------------
# TRANSFORMS
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
# DATASET
# -------------------------
full_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
num_classes = len(full_dataset.classes)

# -------------------------
# TRAIN/VAL SPLIT
# -------------------------
num_samples = len(full_dataset)
indices = list(range(num_samples))
random.shuffle(indices)
split = int(0.8 * num_samples)
train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(datasets.ImageFolder(dataset_dir, transform=val_transform), val_indices)

# -------------------------
# CLASS BALANCING FOR TRAINING
# -------------------------
labels = [full_dataset.samples[i][1] for i in train_indices]
class_sample_count = np.array([labels.count(t) for t in range(num_classes)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in labels])
samples_weight = torch.from_numpy(samples_weight).double()
train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# -------------------------
# DATALOADERS
# -------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# MODEL
# -------------------------
model = models.resnet50(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# -------------------------
# LOSS
# -------------------------
class_weights = torch.tensor(weight, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -------------------------
# TRAIN & VALIDATE FUNCTIONS
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


tta_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

def validate_model(model, dataloader, criterion, use_tta=False):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if use_tta:
                outputs_tta = torch.zeros((inputs.size(0), num_classes)).to(device)
                for _ in range(tta_times):
                    # Apply small TTA perturbations directly on tensors
                    aug_inputs = inputs.clone()
                    if torch.rand(1).item() > 0.5:
                        aug_inputs = torch.flip(aug_inputs, dims=[3])  # horizontal flip
                    aug_inputs = aug_inputs + 0.05 * torch.randn_like(aug_inputs)  # slight noise
                    aug_inputs = torch.clamp(aug_inputs, 0, 1)
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

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# -------------------------
# TRAINING STAGES
# -------------------------
# Stage 1
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
report = classification_report(y_true, y_pred, target_names=full_dataset.classes)
print(report)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (with TTA)")
plt.show()

# -------------------------
# SAVE MODEL
# -------------------------
torch.save(model.state_dict(), "final_resnet50_finetuned.pth")
print("\nModel saved as 'final_resnet50_finetuned.pth'")
