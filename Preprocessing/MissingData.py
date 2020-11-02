import numpy as np
import pandas as pd
import missingno as msno
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def review_data():
    # Read csv file into pandas dataframe
    propertydata = pd.read_csv("propertydata.csv")

    # Print first five rows
    print(propertydata.head())

    # Print whole table
    print(propertydata)

    # Print all columns separately
    print(propertydata['ST_NUM'])
    print(propertydata['ST_NAME'])
    print(propertydata['OWN_OCCUPIED'])
    print(propertydata['NUM_BEDROOMS'])
    print(propertydata['NUM_BATH'])
    print(propertydata['SQ_FT'])

    # Visualize gaps in data
    msno.matrix(propertydata)
    plt.show()
    msno.matrix(propertydata)
    msno.bar(propertydata, color="blue")
    plt.show()


def preprocess_data():
    # Read csv file into pandas dataframe, replace missing values with NaN
    # Handle NUM_BEDROOMS and SQ_FT
    propertydata = pd.read_csv("propertydata.csv", na_values=["na", "--"])

    print(propertydata['ST_NAME'])
    print(propertydata['NUM_BEDROOMS'])
    print(propertydata['SQ_FT'])

    # Handle OWN_OCCUPIED
    cnt = 0
    for row in propertydata['OWN_OCCUPIED']:
        try:
            # Try to cast value to int
            int(row)
            # If possible, replace that value
            propertydata.loc[cnt, 'OWN_OCCUPIED'] = np.nan
        except ValueError:
            pass
        cnt += 1
    print(propertydata['OWN_OCCUPIED'])

    # Handle NUM_BATH
    cnt = 0
    for row in propertydata['NUM_BATH']:
        try:
            # Try to cast value to float
            float(row)
        except ValueError:
            # If NOT possible, replace that value
            propertydata.loc[cnt, 'NUM_BATH'] = np.nan
        cnt += 1
    print(propertydata['NUM_BATH'])

    # Show missing value matrix
    msno.matrix(propertydata)
    plt.show()

    # Simulate listwise deletion
    print(propertydata.dropna())

    # Handle ST_NUM, NUM_BEDROOMS, NUM_BATH, SQ_FT
    # Perform mean imputation and down-cast to int to get rid of values like 2.167 bedrooms
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    propertydata['ST_NUM'] = imp_mean.fit_transform(propertydata['ST_NUM'].values.reshape(-1, 1))
    propertydata['ST_NUM'] = propertydata['ST_NUM'].astype(int)

    propertydata['NUM_BEDROOMS'] = imp_mean.fit_transform(propertydata['NUM_BEDROOMS'].values.reshape(-1, 1))
    propertydata['NUM_BEDROOMS'] = propertydata['NUM_BEDROOMS'].astype(int)

    propertydata['NUM_BATH'] = imp_mean.fit_transform(propertydata['NUM_BATH'].values.reshape(-1, 1))
    propertydata['NUM_BATH'] = propertydata['NUM_BATH'].astype(int)

    propertydata['SQ_FT'] = imp_mean.fit_transform(propertydata['SQ_FT'].values.reshape(-1, 1))
    propertydata['SQ_FT'] = propertydata['SQ_FT'].astype(int)

    print(propertydata['ST_NUM'])
    print(propertydata['ST_NAME'])
    print(propertydata['OWN_OCCUPIED'])
    print(propertydata['NUM_BEDROOMS'])
    print(propertydata['NUM_BATH'])
    print(propertydata['SQ_FT'])

    # Show missing value matrix
    msno.matrix(propertydata)
    plt.show()

    # Simulate listwise deletion
    print(propertydata.dropna())

    # Handle OWN_OCCUPIED
    # Perform most frequent imputation
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    propertydata['OWN_OCCUPIED'] = imp_most_frequent.fit_transform(propertydata['OWN_OCCUPIED'].values.reshape(-1, 1))

    # Show missing value matrix
    msno.matrix(propertydata)
    plt.show()


review_data()
# preprocess_data()
