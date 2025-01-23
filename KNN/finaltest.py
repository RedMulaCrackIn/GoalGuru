import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento del test set
ts = pd.read_csv("test_set.csv", index_col=False)

# Separazione delle feature (X) e del target (y)
ts_x = ts.drop('result', axis=1)
ts_y = ts['result']

# Caricamento del modello KNN pre-addestrato
with open("best_knn_model.pkl", "rb") as f:
    best_model = pickle.load(f)

    # Predizione delle classi sul test set
y_pred = best_model.predict(ts_x)

# Calcolo dell'accuratezza sul test set
test_accuracy = accuracy_score(ts_y, y_pred)

# Stampa dell'accuratezza
print(f"Test accuracy: {test_accuracy}")

# Generazione del report di classificazione
print("\nReport di classificazione:")
print(classification_report(ts_y, y_pred))