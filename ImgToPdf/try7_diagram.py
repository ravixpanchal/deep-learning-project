import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------
# CONFIGURATION
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'D:/RaviKaGroup/ImageProcessing/try1_2110/ImgToPdf/RawOutput'
num_classes = 3
batch_size = 32
num_epochs = 60
lr = 5e-5            # slightly smaller for fine-tuning
cutmix_alpha = 1.0
label_smooth = 0.15
ema_decay = 0.999

# --------------------------
# DATA AUGMENTATION
# --------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
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
# LOAD DATA
# --------------------------
dataset = datasets.ImageFolder(data_dir, transform=train_transform)
class_names = dataset.classes
targets = np.array(dataset.targets)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(np.zeros(len(targets)), targets):
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(
        datasets.ImageFolder(data_dir, transform=val_transform), val_idx)

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(targets), y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)

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
    nn.Dropout(0.5),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)

# Unfreeze deeper layers for fine-tuning
for name, param in model.named_parameters():
    if "features.6" in name or "features.7" in name or "features.8" in name or "features.9" in name:
        param.requires_grad = True

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --------------------------
# EMA setup
# --------------------------
ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
def update_ema(model, ema_state, decay):
    for k, v in model.state_dict().items():
        if k in ema_state:
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.clone().detach(), alpha=1-decay)
            else:
                ema_state[k] = v.clone().detach()

# --------------------------
# CUTMIX IMPLEMENTATION
# --------------------------
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam

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
        update_ema(model, ema_state, ema_decay)

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
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_effnetb0.pth')

print(f"\nBest Validation Accuracy: {best_acc:.2f}%")

# --------------------------
# EVALUATION (EMA + TTA)
# --------------------------
model.load_state_dict(ema_state)
model.eval()

tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
]

def tta_predict(model, img, device):
    preds = []
    for t in tta_transforms:
        inp = t(img).unsqueeze(0).to(device)
        preds.append(torch.softmax(model(inp), dim=1))
    return torch.mean(torch.stack(preds), dim=0)

all_preds, all_labels = [], []
to_pil = transforms.ToPILImage()
with torch.no_grad():
    for inputs, labels in val_loader:
        for i in range(inputs.size(0)):
            pred = tta_predict(model, to_pil(inputs[i].cpu()), device)
            all_preds.append(torch.argmax(pred).cpu().item())
            all_labels.append(labels[i].cpu().item())

# Classification report data
precision = [0.64, 0.17, 0.00]
recall = [0.16, 0.88, 0.00]
f1_score = [0.25, 0.28, 0.00]
classes = ['output_jpg', 'output_jpg2', 'output_jpg3']

# Bar width
bar_width = 0.25
index = np.arange(len(classes))  # x-axis positions for classes

# Plot the Bar Graph
plt.figure(figsize=(10, 6))

plt.bar(index - bar_width, precision, bar_width, label='Precision', color='skyblue')
plt.bar(index, recall, bar_width, label='Recall', color='lightgreen')
plt.bar(index + bar_width, f1_score, bar_width, label='F1-Score', color='salmon')

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.xticks(index, classes)
plt.legend()

plt.tight_layout()
plt.show()

print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
