# Usa una imagen ligera de Python
FROM python:3.10-buster


# Evita problemas de instalación compilando paquetes
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*


# Establece directorio de trabajo
WORKDIR /app

# Copia dependencias y las instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la app
COPY . .

# Expone el puerto que usa Gunicorn
EXPOSE 8080

# Ejecuta la app con Gunicorn
CMD ["gunicorn", "-b", ":8080", "main:app"]
