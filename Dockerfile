FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app/ app/
COPY modelos/ modelos/

# O Cloud Run e o Render informam a porta pela variável PORT
CMD ["sh", "-c", "uvicorn app.main:criar_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
