from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

# Caricamento del test set
ts = pd.read_csv("test_set.csv", index_col=False)
