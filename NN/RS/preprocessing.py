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


# Aggiunta di una colonna 'day' per il giorno della settimana
df['day'] = df['date'].dt.day_name()

# Estrazione dell'ora dalla colonna 'time'
df["hour"] = df["time"].str.replace(":.+", "", regex=True).astype("int")

# Aggiunta di una colonna 'day_code' per il giorno della settimana (0 = Lunedì, 6 = Domenica)
df["day_code"] = df["date"].dt.dayofweek



# Verifica dei duplicati nel dataset
print("\nNumero di duplicati nel dataset:", df.duplicated().sum())

#PULIAMO PRIMA DI STAMPARE
df.formation = df.formation.str.replace("◆", "")
df.formation = df.formation.str.replace("-0", "")

# Analisi delle formazioni più comuni
print("\nConteggio delle formazioni:")
print(df.formation.value_counts())


# Rimozione di caratteri speciali dalla colonna 'formation'
df.formation = df.formation.str.replace("◆", "")
df.formation = df.formation.str.replace("-0", "")

# Categorizzazione delle formazioni meno comuni come "Altro"
value_counts = df.formation.value_counts()
to_replace = value_counts[value_counts < 107].index
df['formation'] = df['formation'].replace(to_replace, 'Altro')

# Verifica delle formazioni dopo la pulizia
print("\nConteggio delle formazioni dopo la pulizia:")
print(df.formation.value_counts())


# Assegnazione dei punti in base al risultato (W = 3, D = 1, L = 0)
df['points'] = df['result'].apply(lambda x: 3 if x == 'W' else 1 if x == 'D' else 0)
df['points'] = df['points'].astype('int')

# Calcolo dei vincitori di ogni stagione
winners = df.groupby(['season', 'team'], observed=False)['points'].sum().reset_index() \
  .sort_values(['season', 'points'], ascending=[True, False]) \
  .groupby('season', observed=False).first()

# Aggiunta della colonna 'season_winner' per indicare il vincitore della stagione
df['season_winner'] = df['season'].map(winners['team'])

# Funzione per gestire i valori mancanti nella colonna 'captain'
def captains_func(data):
    if data['count'] == 0:
        data['count'] = np.nan
    return data


# Conteggio dei capitani per squadra
group = df.groupby('team', observed=False)['captain'].value_counts().reset_index(name='count')
group = group.apply(captains_func, axis=1)
group.dropna(inplace=True)
group = group.drop(columns='count')

# Esempio: Capitani del Liverpool
print("\nCapitani del Liverpool:")
print(group[group['team'] == 'Liverpool'])

# Conversione della colonna 'date' in formato datetime (se non già fatto)
df['date'] = pd.to_datetime(df['date'])

# Ordinamento del dataset per squadra e data
df_sorted = df.sort_values(['team', 'date'])

# Reset dell'indice per riflettere il nuovo ordine
df_sorted = df_sorted.reset_index(drop=True)



# Funzione per verificare l'ordinamento corretto
def verify_sorting(data):
    is_sorted = data.groupby('team', observed=False)['date'].is_monotonic_increasing.all()
    if is_sorted:
        print("Data is correctly sorted by date for each team.")
    else:
        print("WARNING: Data is not correctly sorted. Please check for inconsistencies.")

# Verifica dell'ordinamento
verify_sorting(df_sorted)

# Conversione delle colonne numeriche in tipo float
num_cols = ['sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'xga', 'xg', 'gf', 'ga']
for col in num_cols:
    df_sorted[col] = pd.to_numeric(df_sorted[col])

# Funzione per calcolare le metriche avanzate
def calculate_fk_pk_ratios(data):
    data['fk_ratio'] = data['fk'] / data['sh']
    data['pk_conversion_rate'] = data['pk'] / data['pkatt']
    data['pk_per_shot'] = data['pkatt'] / data['sh']

    # Gestione dei valori infiniti
    data['fk_ratio'] = data['fk_ratio'].replace([np.inf, -np.inf], np.nan)
    data['pk_conversion_rate'] = data['pk_conversion_rate'].replace([np.inf, -np.inf], np.nan)
    data['pk_per_shot'] = data['pk_per_shot'].replace([np.inf, -np.inf], np.nan)

    # Conversione in percentuali
    data['fk_percentage'] = data['fk_ratio'] * 100
    data['pk_conversion_percentage'] = data['pk_conversion_rate'] * 100
    data['pk_per_shot_percentage'] = data['pk_per_shot'] * 100

    return data

# Applicazione della funzione
df_sorted = calculate_fk_pk_ratios(df_sorted)

# Rimozione delle colonne non necessarie
df_sorted.drop(['pk_conversion_rate', 'pk_conversion_percentage'], axis=1, inplace=True)
