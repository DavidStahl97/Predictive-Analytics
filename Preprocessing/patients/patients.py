# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Preprocessing

# %%
import numpy as np
import pandas as pd
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import os

# %% [markdown]
# ## Inspect Data

# %%
directory = os.getcwd()


# %%
data = pd.read_csv('patientdata.csv', na_values=["na", "--", "unknown"], index_col='patient_id')
data


# %%
msno.matrix(data)

# %% [markdown]
# ## Handling Missing Data

# %%

def missing_mean(column_name, missing_data):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    missing_data[column_name] = imp_mean.fit_transform(missing_data[column_name].values.reshape(-1, 1)).astype(int)

def missing_constant(constant, column_name, missing_data):
    imp_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=constant)
    missing_data[column_name] = imp_constant.fit_transform(missing_data[column_name].values.reshape(-1, 1))

# city
data = data.drop('city', axis=1)

# gender 
data = data.replace({'gender': {'male': 'm', 'female': 'f'}})
data = data[(data['gender'] == 'm') | (data['gender'] == 'f')]

# age_yrs
missing_mean('age_yrs', data)

# weight_kg
missing_mean('weight_kg', data)

# height_cm
missing_mean('height_cm', data)

# diabetes
missing_constant('no', 'diabetes', data)

# heart_disease
missing_constant('no', 'heart_disease', data)

# lung_disease
missing_constant('no', 'lung_disease', data)

msno.matrix(data)

# %% [markdown]
# ## OneHotEncoder

# %%

def encode_strings(column_name, data):    
    label_encoder = LabelEncoder()
    data[column_name] = label_encoder.fit_transform(data[column_name])

# gender
encode_strings('gender', data)    

# diabetes
encode_strings('diabetes', data)

# heart_disease
encode_strings('heart_disease', data)

# lung_disease
encode_strings('lung_disease', data)

data

# %% [markdown]
# ## Scaling

# %%
data_scaled = preprocessing.scale(data)
df = pd.DataFrame(data_scaled, 
        index=data.index,
        columns=data.columns)
df

# %% [markdown]
# ## Export to Excel

# %%
df.to_excel('patients.xlsx')

