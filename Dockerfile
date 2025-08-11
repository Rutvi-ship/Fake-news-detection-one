# ---- Base ----
FROM python:3.11-slim

# Faster Python, cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (enables Docker layer caching)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app (app.py + joblib files + anything else)
COPY . .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Start the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
