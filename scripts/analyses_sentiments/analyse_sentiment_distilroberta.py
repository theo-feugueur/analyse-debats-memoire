import pandas as pd
import os
from transformers import pipeline, AutoTokenizer
import re

# 🚀 Chargement des modèles NLP
# Modèle d'émotions : j-hartmann/emotion-english-distilroberta-base
print("\U0001F680 Chargement des modèles NLP...")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Charger le tokenizer associé au modèle
tokenizer_emotion = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# 📚 Chemins des fichiers
input_file = "data/raw/debat_30_anniversaire.csv"  # Fichier d'entrée (déjà traduit)
output_file = "data/processed/debat_30_anniversaire_avec_sentiment.csv"

# 🛠️ Suppression de l'ancien fichier de sortie si existant
if os.path.exists(output_file):
    try:
        os.remove(output_file)
    except PermissionError:
        print(f"\u274C Impossible d'écrire sur {output_file}. Fermez le fichier s'il est ouvert et réessayez.")
        exit()

# 📃 Charger le fichier CSV
df = pd.read_csv(input_file, sep=",", encoding="utf-8")

# Fonction pour segmenter un texte en blocs intelligents
def segment_text(text, tokenizer, max_tokens=512):
    """Segmente le texte en respectant les phrases et la limite de 512 tokens."""
    sentences = re.split(r'(?<=[.!?]) +', text)  # Coupe aux fins de phrase
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        tokens = tokenizer.encode(current_segment + " " + sentence, add_special_tokens=True)
        if len(tokens) <= max_tokens:
            current_segment += " " + sentence
        else:
            segments.append(current_segment.strip())
            current_segment = sentence
    
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

# Fonction pour analyser les émotions avec agrégation
def get_emotions(text):
    """Analyse les émotions et cumule les scores sur tous les segments."""
    segments = segment_text(text, tokenizer_emotion)
    emotion_scores = {}
    
    for segment in segments:
        results = emotion_analyzer(segment)
        for res in results:
            for entry in res:
                label, score = entry['label'], entry['score']
                emotion_scores[label] = emotion_scores.get(label, 0) + score
    
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{emo} ({round(score, 2)})" for emo, score in sorted_emotions]) if sorted_emotions else "Neutral"

# ⏳ Analyse des sentiments en cours...
print("\u23F3 Analyse des sentiments en cours...")

# Test visuel sur la première intervention
first_intervention = df["intervention"].iloc[0]
print("\n\U0001F4A1 Exemple sur la première intervention :")
print(first_intervention)
print("Segments d'analyse :", segment_text(first_intervention, tokenizer_emotion))
print("\U0001F52E Résultat des émotions :", get_emotions(first_intervention))

# Application sur toutes les interventions
df["emotions"] = df["intervention"].apply(get_emotions)

print("\u2705 Analyse terminée !")

# 📁 Sauvegarde du fichier avec les nouvelles colonnes
df.to_csv(output_file, sep=",", encoding="utf-8", index=False)
print(f"\U0001F4C1 Fichier analysé enregistré sous : {output_file}")
