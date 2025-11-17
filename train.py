import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_DIR = "dataset"
BATCH = 32
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(DATA_DIR + "/train", transform=train_transform)
valid_data = datasets.ImageFolder(DATA_DIR + "/valid", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH)

# ---- MobileNetV2 ----
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for img, lbl in train_loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)

        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("Saved as model.pth")
