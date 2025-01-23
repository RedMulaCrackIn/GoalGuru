import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import itertools


# Caricamento del dataset pre-processato
tr = pd.read_csv("matches_final.csv", index_col=False)

# Separazione delle feature (X) e del target (y)
X = tr.drop('result', axis=1)
y = tr['result']

# Identificazione delle colonne categoriche e numeriche
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# One-Hot Encoding per le variabili categoriche
X_categorical_encoded = pd.get_dummies(X[categorical_cols], columns=categorical_cols)

# Standardizzazione delle variabili numeriche
scaler = StandardScaler()
X_numerical_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)
