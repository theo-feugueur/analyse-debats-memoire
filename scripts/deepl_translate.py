import pandas as pd
import deepl

# Clé API DeepL (⚠️ Ne pas partager publiquement)
DEEPL_API_KEY = "c77aef13-79e9-43f5-8036-49b16b6e5e07:fx"

# Définir les fichiers
input_file = "data/raw/debat_30_anniversaire.csv"  # Fichier original
output_file = "data/processed/debat_30_anniversaire_translated.csv"  # Fichier traduit

# Charger le fichier CSV
df = pd.read_csv(input_file, sep=",", encoding="utf-8")

# Initialiser le traducteur DeepL
translator = deepl.Translator(DEEPL_API_KEY)

# Fonction pour traduire un texte
def translate_text(text):
    try:
        return translator.translate_text(text, target_lang="EN-US").text
    except Exception as e:
        print(f"❌ Erreur de traduction : {e}")
        return text  # Si erreur, on garde le texte original

# Appliquer la traduction à la colonne "intervention"
print("⏳ Traduction en cours...")
df["intervention_traduite"] = df["intervention"].apply(translate_text)
print("✅ Traduction terminée !")

# Sauvegarder le fichier traduit
df.to_csv(output_file, sep=",", encoding="utf-8", index=False)
print(f"📁 Fichier traduit enregistré sous : {output_file}")
