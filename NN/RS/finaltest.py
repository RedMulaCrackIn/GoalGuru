from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

# Caricamento del test set
ts = pd.read_csv("test_set.csv", index_col=False)

# Separazione delle feature (X) e del target (y)
ts_x = ts.drop('result', axis=1)
ts_y = ts['result']

# Caricamento del modello pre-addestrato
best_model = load_model('best_model.h5')

# Visualizzazione del riepilogo del modello
print("Riepilogo del modello:")
print(best_model.summary())

# Valutazione del modello sul test set
test_loss, test_accuracy = best_model.evaluate(ts_x, ts_y, verbose=1)

# Stampa della loss e dell'accuratezza
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Predizione delle classi sul test set
y_pred = best_model.predict(ts_x)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generazione del report di classificazione
print("\nReport di classificazione:")
print(classification_report(ts_y, y_pred_classes))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calcolo della matrice di confusione
conf_matrix = confusion_matrix(ts_y, y_pred_classes)

# Visualizzazione della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Loss', 'Draw', 'Win'],
            yticklabels=['Loss', 'Draw', 'Win'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matrice di Confusione')
# Salva la figura in un file
plt.savefig('matrice_di_confusione.png')
plt.show()

# Salvataggio dei risultati in un file di testo
with open("test_results.txt", "w") as f:
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")
    f.write("\nReport di classificazione:\n")
    f.write(classification_report(ts_y, y_pred_classes))