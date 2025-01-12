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