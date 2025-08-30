import argparse, json, os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

# -------------------------
# Transforms
# -------------------------
def get_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tfms, val_tfms

# -------------------------
# Train & Eval functions
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return running_loss / total, correct / total, all_preds, all_labels

# -------------------------
# Build model
# -------------------------
def build_model(num_classes, arch="efficientnet_b0", pretrained=True):
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported architecture")
    return model

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tfms, val_tfms = get_transforms(args.img_size)

    # Dataset
    full_ds = datasets.ImageFolder(args.data_dir, transform=train_tfms)
    n = len(full_ds)
    val_n = max(1, int(0.15 * n))
    train_n = n - val_n
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_n, val_n])
    val_ds.dataset.transform = val_tfms

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    labels = full_ds.classes
    os.makedirs("models", exist_ok=True)
    with open("models/labels.json", "w") as f:
        json.dump({i: name for i, name in enumerate(labels)}, f, indent=2)

    model = build_model(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_path = "models/model.pth"
    patience, bad = 5, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(), "arch": "efficientnet_b0", "num_classes": len(labels)}, best_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Load best model and show classification report
    ckpt = torch.load(best_path, map_location=device)
    model = build_model(num_classes=ckpt["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    _, _, preds, gts = evaluate(model, val_loader, criterion, device)
    try:
        print(classification_report(gts, preds, target_names=labels, zero_division=0))
    except Exception as e:
        print("Classification report unavailable:", e)

if __name__ == "__main__":
    main()
