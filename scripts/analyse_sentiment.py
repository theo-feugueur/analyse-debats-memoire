import pandas as pd
import os
from transformers import pipeline, AutoTokenizer

# 🚀 Chargement des modèles NLP
print("🚀 Chargement des modèles NLP...")
polarite_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=5)

# Charger les tokenizers associés aux modèles
tokenizer_polarite = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer_emotion = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

# Chemins des fichiers
input_file = "data/raw/debat_30_anniversaire.csv"  # Fichier traduit
output_file = "data/processed/debat_30_anniversaire_avec_sentiment.csv"

# Charger le fichier CSV
df = pd.read_csv(input_file, sep=",", encoding="utf-8")

# Vérifier et supprimer le fichier de sortie s'il existe déjà
if os.path.exists(output_file):
    try:
        os.remove(output_file)
    except PermissionError:
        print(f"❌ Impossible d'écrire sur {output_file}. Fermez le fichier s'il est ouvert et réessayez.")
        exit()

# Fonction pour segmenter un texte en blocs stricts de 512 tokens
def segment_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, max_length=max_tokens, return_tensors="pt")
    token_chunks = tokens['input_ids'][0].split(max_tokens - 2)
    segments = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]
    return segments

# Fonction pour analyser la polarité d'une intervention segmentée
def get_polarity(text):
    segments = segment_text(text, tokenizer_polarite)
    scores = []
    
    for segment in segments:
        result = polarite_analyzer(segment)[0]
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    return "POSITIVE" if avg_score > 0.1 else "NEGATIVE" if avg_score < -0.1 else "NEUTRAL"

# Fonction pour analyser les émotions avec une pondération améliorée
def get_emotions(text, seuil=0.1):
    segments = segment_text(text, tokenizer_emotion)
    emotion_scores = {}
    
    for segment in segments:
        results = emotion_analyzer(segment)
        for res in results:
            for emo in res:
                if emo['score'] > seuil:
                    emotion_scores[emo['label']] = emotion_scores.get(emo['label'], []) + [emo['score']]
    
    # Pondération des émotions par fréquence et score moyen
    aggregated_emotions = {emo: sum(scores) / len(scores) for emo, scores in emotion_scores.items()}
    sorted_emotions = sorted(aggregated_emotions.items(), key=lambda x: x[1], reverse=True)
    
    return ", ".join([f"{emo} ({round(score, 2)})" for emo, score in sorted_emotions[:5]]) if sorted_emotions else "Neutral (0.85)"

# Appliquer l'analyse sur chaque intervention
print("⏳ Analyse des sentiments en cours...")
df["polarite"] = df["intervention"].apply(get_polarity)
df["emotions"] = df["intervention"].apply(lambda x: get_emotions(x, seuil=0.1))
print("✅ Analyse terminée !")

# Sauvegarder le fichier avec les nouvelles colonnes
df.to_csv(output_file, sep=",", encoding="utf-8", index=False)
print(f"📁 Fichier analysé enregistré sous : {output_file}")
