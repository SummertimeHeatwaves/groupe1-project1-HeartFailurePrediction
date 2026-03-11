Construire l'image
docker build -t heart-failure .

# Lancer le container
docker run -p 8501:8501 heart-failure

# → Ouvrir http://localhost:8501