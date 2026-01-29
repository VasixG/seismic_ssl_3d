FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements_frontend.txt /app/
RUN pip install --no-cache-dir -r requirements_frontend.txt

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.maxUploadSize=2048"]
