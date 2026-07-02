"""FastAPI-сервис инференса.

Запуск:  uvicorn app.main:app --reload
Docs:    http://localhost:8000/docs
"""
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.model import CLASSES, LABELS_RU, load_model, predict

app = FastAPI(
    title="Apple Leaf Disease API",
    description="Классификация болезней листьев яблони (PyTorch)",
    version="1.0.0",
)

model = load_model()  # MODEL_NAME из env, по умолчанию 'resnet'


@app.get("/health")
def health():
    return {"status": "ok", "model": getattr(model, "_model_name", "?"), "classes": CLASSES}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Ожидается файл-изображение")
    data = await file.read()
    try:
        probs = predict(model, data)
    except Exception as e:  # noqa: BLE001 — вернём читаемую ошибку клиенту
        raise HTTPException(status_code=422, detail=f"Не удалось обработать изображение: {e}")

    top = max(probs, key=probs.get)
    return {
        "prediction": top,
        "label_ru": LABELS_RU.get(top, top),
        "confidence": probs[top],
        "probabilities": probs,
    }
