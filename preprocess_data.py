#!/usr/bin/env python

"""
Prepare datasets for training and testing
"""

from IPython import embed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Subset of the columns to use
cols = ['Month', 'DayOfWeek', 'Distance',
        'DepDelay', 'CRSDepTime', 'UniqueCarrier', 'Origin', 'Dest']
categorical_cols = ['UniqueCarrier', 'Origin', 'Dest']


def _get_df_from_file(file_name, n):
    df = pd.read_csv(file_name, usecols=cols)
    df = df.dropna(subset=cols)
    # Keep `n` samples
    df = df.sample(n=n, random_state=42)

    # Create binary labels from the delay column, and delete it from the
    # training data
    labels = df.DepDelay > 15
    del df['DepDelay']

    # Discard minutes in departure times
    df.CRSDepTime = df.CRSDepTime // 100    
    return df, labels


if __name__ == '__main__':
    df2006, y2006 = _get_df_from_file('2006.csv', 100*1000)
    df2007, y2007 = _get_df_from_file('2007.csv', 200*1000)

    # xgboost wants numbers for categorical variables
    for col in categorical_cols:
        lenc = LabelEncoder().fit(pd.concat([df2006[col], df2007[col]]))
        df2006[col] = lenc.transform(df2006[col])
        df2007[col] = lenc.transform(df2007[col])

    # Get test/validation sets from 2007 data
    X_val, X_test, y_val, y_test = train_test_split(
        df2007, y2007, test_size=0.5, random_state=43)

    data = dict(
        feature_names=df2006.columns,
        X_train=df2006, y_train=y2006,
        X_test=X_test, y_test=y_test,
        X_val=X_val, y_val=y_val,
    )

    for k, v in data.items():
        print(k, v.shape)

    np.savez('airlines_data.npz', **data)
