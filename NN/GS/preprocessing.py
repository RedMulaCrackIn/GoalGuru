import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione delle opzioni di visualizzazione per i DataFrame
pd.set_option('display.max_columns', None)

# Ignorare i warning per mantenere l'output pulito
import warnings
warnings.filterwarnings('ignore')

# Caricamento del dataset
df = pd.read_csv('matches.csv')

# Visualizzazione delle prime righe del dataset
print("Prime righe del dataset:")
print(df.head())

# Statistiche descrittive del dataset
print("\nStatistiche descrittive:")
print(df.describe())

# Informazioni generali sul dataset (tipi di dati, valori non nulli)
print("\nInformazioni sul dataset:")
print(df.info())

# Rimozione delle colonne non necessarie
df.drop(columns=["Unnamed: 0", "comp", "round", "attendance", "match report", "notes"], inplace=True)

# Conversione della colonna 'date' in formato datetime
df["date"] = pd.to_datetime(df["date"])

# Conversione di alcune colonne in tipo 'category' per ottimizzare la memoria
df['venue'] = df['venue'].astype('category')
df['opponent'] = df['opponent'].astype('category')
df['team'] = df['team'].astype('category')
df['result'] = df['result'].astype('category')
