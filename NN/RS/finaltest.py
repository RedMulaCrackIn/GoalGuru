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

