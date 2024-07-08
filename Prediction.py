import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Read CSV file
    df = pd.read_csv('gld_price_data.csv')
    features = df.drop(['Date', 'GLD'], axis=1).values.tolist()
    values = np.array(df['GLD'].tolist())

    X_train, X_test, Y_train, Y_test = train_test_split(features, values, test_size=0.4)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    print(f"Mean Squared Error: {mean_squared_error(Y_test, predictions)}")
    print(f"Mean Absolute Error: {mean_absolute_error(Y_test, predictions)}")
    print(f"R² Score: {r2_score(Y_test, predictions)}")
    
if __name__ == "__main__":
    main()