# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port par défaut de Streamlit
EXPOSE 8503

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8503", "--server.address=0.0.0.0"]