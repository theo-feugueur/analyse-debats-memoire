import pandas as pd  # Importer pandas pour manipuler le CSV

# Définir le chemin du fichier CSV
file_path = "data/raw/debat_30_anniversaire.csv"

try:
    # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")  # Change `sep=";"` si nécessaire

    # Afficher un message de confirmation
    print("✅ Fichier chargé avec succès !")
    
    # Afficher les 5 premières lignes du fichier
    print(df.head())

except FileNotFoundError:
    print("❌ Erreur : Fichier non trouvé ! Vérifie le chemin.")

except Exception as e:
    print(f"❌ Une erreur est survenue : {e}")
# Afficher des informations générales sur le fichier
print("\n📊 Informations sur le fichier CSV :")
print(df.info())  # Donne le nombre de lignes, de colonnes et les types de données

print("\n🔍 Aperçu des valeurs manquantes :")
print(df.isnull().sum())  # Affiche le nombre de valeurs manquantes par colonne

print("\n📈 Statistiques générales :")
print(df.describe(include="all"))  # Donne un résumé statistique des colonnes
