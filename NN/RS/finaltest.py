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