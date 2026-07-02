import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models


class AppleLeaf(data.Dataset):
    def __init__(self, root, trans=None):
        self.root = root
        self.transform = trans

        self.classes = sorted(d for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, img_name)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, label = self.samples[index]
        img = Image.open(img).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = AppleLeaf(root='Apple_Disease_Dataset/train', trans=transform)
test_dataset = AppleLeaf(root='Apple_Disease_Dataset/test', trans=transform)
val_dataset = AppleLeaf(root='Apple_Disease_Dataset/val', trans=transform)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(64, 128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(128, 256, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(256, 512, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2), )

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512 * 7 * 7, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.4),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.4),
                                        nn.Linear(512, len(train_dataset.classes)), )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

while True:
    choice = input("Выберите тип модели(1/2): 1.custom\n2.resnet\n")
    if choice in ('1', '2'):
        MODEL_TYPE = int(choice)
        break
    print("Неверный ввод, попробуй снова")

MODEL_NAME= 'custom' if MODEL_TYPE == 1 else 'resnet'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
load_path = f'Apple_Leaf/model_{MODEL_NAME}.pth'
model_loaded = os.path.exists(load_path)

if MODEL_TYPE == 1:
    model = Net()
else:
    # ImageNet-веса нужны только для обучения с нуля; если есть свои — не качаем
    weights = None if model_loaded else 'IMAGENET1K_V1'
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

if model_loaded:
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f'Загружены веса: {load_path}')


criterion = nn.CrossEntropyLoss()
model = model.to(device)
if not model_loaded:
    opt = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        t_l = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}',leave=False)
        for x_batch, y_batch in t_l:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            t_l.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs} — Avg Loss: {avg_loss:.4f}')

        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                prediction = model(x_batch)
                loss = criterion(prediction, y_batch)
                val_loss += loss.item()
                total += y_batch.size(0)
                correct += (prediction.argmax(1) == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100
        print(
            f'Epoch {epoch + 1}/{epochs} - Train_loss: {avg_loss:.4f} - Avg_val_loss: {avg_val_loss:.4f} - Avg_val_acc: {val_acc:.2f}%')


    os.makedirs('Apple_Leaf', exist_ok=True)
    torch.save(model.state_dict(), f'Apple_Leaf/model_{MODEL_NAME}.pth')


model.eval()
correct = 0
total = 0
test_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)

        test_loss += loss.item()
        preds = prediction.argmax(1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_acc = (correct / total) * 100
print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
print(confusion_matrix(all_labels, all_preds))
