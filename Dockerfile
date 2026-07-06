# Инференс-сервис Apple Leaf Disease (Gradio UI + FastAPI).
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch — образ меньше и не требует GPU в контейнере.
ENV PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=resnet

# Сначала зависимости — кэшируется, пока не меняется requirements-app.txt.
COPY requirements-app.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision==0.27.1 \
    && pip install fastapi "uvicorn[standard]" python-multipart gradio \
       pillow==12.3.0 numpy==2.5.0

# Код и веса.
COPY app/ ./app/
COPY Apple_Leaf/ ./Apple_Leaf/

EXPOSE 7860 8000

# По умолчанию поднимаем Gradio-интерфейс.
CMD ["python", "-m", "app.gradio_app"]
