import io
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

LABELS_RU = {
    "Apple___Apple_scab": "Парша яблони",
    "Apple___Black_rot": "Чёрная гниль",
    "Apple___Cedar_apple_rust": "Ржавчина",
    "Apple___healthy": "Здоровый лист",
}

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
else "cpu")


class Net(nn.Module):

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding="same"), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.Dropout(p=0.4),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name):
    if model_name == "custom":
        return Net()
    if model_name == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        return model
    raise ValueError(f"Unknown model_name: {model_name!r} (expected 'custom'/'resnet')")


def gradcam_layer(model, model_name):
    if model_name == "custom":
        return model.features[12]  # Conv2d(256, 512) → карта 14x14
    return model.layer4[-1]  # последний BasicBlock ResNet18


def load_model(model_name=None, weights_dir="Apple_Leaf"):
    model_name = model_name or os.environ.get("MODEL_NAME", "resnet")
    path = os.path.join(weights_dir, f"model_{model_name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найдены веса: {path}. Обучите модель через apple_leaf.py "
            f"или положите файл model_{model_name}.pth в {weights_dir}/."
        )
    model = build_model(model_name)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    model._model_name = model_name  # пригодится для Grad-CAM
    return model


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")


@torch.no_grad()
def predict(model, image):
    pil = _to_pil(image)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return {cls: float(p) for cls, p in zip(CLASSES, probs)}


def _jet(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam(model, image, target_class=None, alpha=0.5):
    model_name = getattr(model, "_model_name", "resnet")
    layer = gradcam_layer(model, model_name)

    pil = _to_pil(image)
    x = transform(pil).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    activations, gradients = {}, {}

    def fwd_hook(_m, _inp, out):
        activations["value"] = out

    def bwd_hook(_m, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        if target_class is None:
            target_class = int(logits.argmax(1).item())
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        acts = activations["value"][0]  # (C, h, w)
        grads = gradients["value"][0]  # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # усреднение градиентов по пространству
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
    finally:
        h1.remove()
        h2.remove()

    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(pil.size, Image.BILINEAR)
    heat = _jet(np.asarray(cam_img) / 255.0)  # HxWx3 в [0, 1]
    heat = Image.fromarray(np.uint8(heat * 255))
    overlay = Image.blend(pil, heat, alpha=alpha)
    return overlay, CLASSES[target_class]
