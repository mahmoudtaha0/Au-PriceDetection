import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read CSV file
df = pd.read_csv('gld_price_data.csv')
features = df.drop(['Date', 'GLD'], axis=1).values.tolist()
values = np.array(df['GLD'].tolist())