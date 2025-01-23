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

# Unione delle variabili categoriche codificate e numeriche standardizzate
X_categorical_encoded = X_categorical_encoded.reset_index(drop=True)
X_numerical_scaled = X_numerical_scaled.reset_index(drop=True)
X_final = pd.concat([X_categorical_encoded, X_numerical_scaled], axis=1)

# Codifica del target (y) in valori numerici
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Creazione del test set (ultima stagione)
test_x = X_final.tail(761)
test_y = y_encoded[-761:]
test_x['result'] = test_y
test_x.to_csv("test_set.csv", index=False)

# Rimozione dell'ultima stagione dal training set
X_final = X_final.iloc[:-761]
y_encoded = y_encoded[:-761]

# Divisione in training e validation set
X_train, X_val, y_train, y_val = train_test_split(X_final, y_encoded, test_size=0.2, random_state=42)

# Griglia di iperparametri per KNN
GRID_SEARCH = {
    "n_neighbors": [3, 5, 7, 9, 11],  # Numero di vicini
    "weights": ['uniform', 'distance'],  # Peso dei vicini
    "metric": ['euclidean', 'manhattan']  # Metrica di distanza
}

# Creazione di tutte le combinazioni di iperparametri
grid_combinations = list(itertools.product(
    GRID_SEARCH['n_neighbors'],
    GRID_SEARCH['weights'],
    GRID_SEARCH['metric']
))

# Variabili per tenere traccia dei migliori iperparametri
best_params = None
best_val_accuracy = 0

# Lista per memorizzare le accuracy durante l'addestramento
val_accuracies = []

# Ciclo per testare ogni combinazione di iperparametri
for combination in grid_combinations:
    n_neighbors, weights, metric = combination

    print(f"Testing combination: n_neighbors={n_neighbors}, weights={weights}, metric={metric}")

    # Creazione del modello KNN con i parametri correnti
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    # Addestramento del modello
    model.fit(X_train, y_train)

    # Valutazione dell'accuratezza sul validation set
    val_accuracy = model.score(X_val, y_val)

    # Aggiungi l'accuratezza alla lista
    val_accuracies.append(val_accuracy)

    print(f"Validation accuracy: {val_accuracy}")

    # Aggiornamento dei migliori iperparametri
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "metric": metric
        }

# Stampa dei migliori iperparametri
print("Best hyperparameters found:")
print(best_params)
print(f"Best validation accuracy: {best_val_accuracy}")



# Creazione del modello KNN con i migliori iperparametri
best_model = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"],
    weights=best_params["weights"],
    metric=best_params["metric"]
)

# Addestramento del modello
best_model.fit(X_train, y_train)

# Valutazione del modello sul validation set
val_predictions = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

print(f"Final validation accuracy with best hyperparameters: {val_accuracy}")

# Salvataggio del modello
with open("best_knn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Plot dell'accuratezza durante la ricerca degli iperparametri
plt.figure(figsize=(8, 6))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy during Hyperparameter Search')
plt.xlabel('Combination Index')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('validation_accuracy.png')
plt.show()
