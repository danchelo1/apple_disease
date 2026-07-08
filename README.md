# 🍏 Apple Leaf Disease Classification

Классификация болезней листьев яблони с помощью сверточных нейронных сетей на PyTorch. Проект поддерживает два подхода на выбор: собственная CNN, написанная с нуля, и transfer learning на базе ResNet18.

## 📋 Описание

Модель определяет одно из четырёх состояний листа яблони по фотографии:

| Класс | Описание |
|---|---|
| `Apple___Apple_scab` | Парша яблони |
| `Apple___Black_rot` | Чёрная гниль |
| `Apple___Cedar_apple_rust` | Ржавчина |
| `Apple___healthy` | Здоровый лист |

## 📊 Датасет

[Apple Disease Dataset](https://www.kaggle.com/datasets/showravdhar/apple-disease-dataset) — 9714 изображений, разбитых на train/val/test, разложенных по папкам-классам.

Структура:
```
Apple_Disease_Dataset/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___Cedar_apple_rust/
│   └── Apple___healthy/
├── val/
│   └── (та же структура)
└── test/
    └── (та же структура)
```

## 🧠 Архитектуры

### 1. Custom CNN

Свёрточная сеть, написанная с нуля:

- 5 свёрточных блоков: `Conv2d → ReLU → MaxPool2d`, каналы растут 3 → 32 → 64 → 128 → 256 → 512
- Классификатор: `Flatten → Linear(1024) → ReLU → Dropout → Linear(512) → ReLU → Dropout → Linear(num_classes)`

### 2. ResNet18 (Transfer Learning)

Предобученная на ImageNet сеть с замороженными весами, дообучается только последний полносвязный слой (`model.fc`) под 4 класса.

## 📈 Результаты

| Модель | Test Accuracy | Test Loss |
|---|---------------|-----------|
| Custom CNN | 98.87%        | 0.0384    |
| ResNet18 (transfer learning) | 99.07%        | 0.0328    |

### Classification Report (Custom CNN)

```
                          precision    recall  f1-score   support

     Apple___Apple_scab       0.97      0.98      0.98       504
      Apple___Black_rot       1.00      0.99      1.00       497
Apple___Cedar_apple_rust       1.00      0.98      0.99       440
       Apple___healthy       0.98      1.00      0.99       502

               accuracy                           0.99      1943
              macro avg       0.99      0.99      0.99      1943
           weighted avg       0.99      0.99      0.99      1943
```

## 🚀 Как запустить

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Подготовка датасета

Скачай [датасет с Kaggle](https://www.kaggle.com/datasets/showravdhar/apple-disease-dataset) и распакуй в корень проекта так, чтобы получилась папка `Apple_Disease_Dataset/` со структурой train/val/test.

### Запуск обучения

```bash
python apple_leaf.py
```

При запуске скрипт спросит, какую модель использовать:

```
Выберите тип модели(1/2): 1.custom
2.resnet
```

Если веса модели уже сохранены в `Apple_Leaf/model_custom.pth` или `Apple_Leaf/model_resnet.pth` — обучение пропускается, и сразу происходит загрузка модели с последующим тестированием.

## 🌐 Инференс-сервис (деплой)

Обученную модель можно запустить как готовый веб-сервис. Инференс-код (`app/`) не зависит от датасета и тренировочного скрипта: нужны только веса в `Apple_Leaf/`.

```bash
pip install -r requirements-app.txt
python -m app.gradio_app
```

Откроется на http://localhost:7860 — загружаешь фото листа и получаешь диагноз + **Grad-CAM** (визуализацию зон, на которые «смотрит» сеть).

По умолчанию используется ResNet18. Сменить модель — переменной окружения `MODEL_NAME=custom`.

## 🛠 Технологии

**Обучение**
- Python 3
- PyTorch / torchvision
- scikit-learn (метрики: classification report, confusion matrix)
- tqdm (прогресс-бар обучения)
- Pillow (обработка изображений)

**Инференс-сервис**
- Gradio (веб-интерфейс)
- NumPy (Grad-CAM / визуализация зон внимания)

## 📁 Структура проекта

```
.
├── apple_leaf.py           # обучение: датасет, модели, тренировка, тест
├── app/                    # инференс-сервис
│   ├── model.py            # архитектуры, загрузка весов, препроцессинг, Grad-CAM
│   └── gradio_app.py       # веб-интерфейс
├── Apple_Disease_Dataset/  # датасет (не в репозитории, см. .gitignore)
├── Apple_Leaf/             # сохранённые веса моделей (не в репозитории)
├── requirements.txt        # зависимости для обучения
├── requirements-app.txt    # зависимости для инференс-сервиса
└── README.md
```

## 💡 Возможные улучшения

- Аугментации данных (повороты, отражения, изменение яркости) для лучшей генерализации на реальных полевых фото
- Разморозка дополнительных слоёв ResNet (`layer4`) для более глубокого дообучения
